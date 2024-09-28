import torch
import math
import torch.nn as nn


class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""

    def __init__(self, *, s_n, t_n):
        super(SRRL, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, cls_t):
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)

        pred_feat_s = cls_t(trans_feat_s)

        return trans_feat_s, pred_feat_s

