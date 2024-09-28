from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, weight):
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

        for i in range(num_stu):
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)

        loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)
        return loss


class SelfA(nn.Module):
    """Cross-layer Self Attention"""

    def __init__(self, feat_dim, s_n, t_n, soft, factor=4):
        super(SelfA, self).__init__()

        self.soft = soft
        self.s_len = len(s_n)
        self.t_len = len(t_n)
        self.feat_dim = feat_dim

        # query and key mapping
        for i in range(self.s_len):
            setattr(self, 'query_' + str(i), MLPEmbed(feat_dim, feat_dim // factor))
        for i in range(self.t_len):
            setattr(self, 'key_' + str(i), MLPEmbed(feat_dim, feat_dim // factor))

        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, 'regressor' + str(i) + str(j), Proj(s_n[i], t_n[j]))  # there exists 16 regressors

    def forward(self, feat_s, feat_t):

        sim_s = list(range(self.s_len))  # sim_s = [0, 1, 2, 3]
        sim_t = list(range(self.t_len))  # sim_t = [0, 1, 2, 3]
        bsz = self.feat_dim

        # similarity matrix
        for i in range(self.s_len):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())  # sim_s = {list4}[tensor(64, 64)]
        for i in range(self.t_len):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())

        # calculate student query
        proj_query = self.query_0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, self.s_len):
            temp_proj_query = getattr(self, 'query_' + str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)  # proj_query = {tensor:(64, 4, 16)}

        # calculate teacher key
        proj_key = self.key_0(sim_t[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, self.t_len):
            temp_proj_key = getattr(self, 'key_' + str(i))(sim_t[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)  # proj_key = {tensor:(64, 16, 4)}

        # attention weight: batch_size X No. stu feature X No.tea feature
        energy = torch.bmm(proj_query, proj_key) / self.soft  # energy:{tensor:(64, 4, 4)}
        attention = F.softmax(energy, dim=-1)

        # feature dimension alignment
        proj_value_stu = []
        value_tea = []
        for i in range(self.s_len):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(self.t_len):
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    target = feat_t[j]
                elif s_H <= t_H:
                    source = feat_s[i]
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))

                proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(source))
                value_tea[i].append(target)

        return proj_value_stu, value_tea, attention  # proj_value_stu = {list:4}[list:4]  value_tea = {list:4}[list:4]


class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear mapping for attention calculation"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_out),
            self.l2norm,
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim_out, dim_out),
            self.l2norm,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))

        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out