"""
VGG for RepKD.
you can equip extra branch to vgg during training,
and equivalent convert the trained model into vanilla vgg8
(c) Yujie Zheng
"""
import torch.nn as nn
import torch.nn.functional as F
import math
from .convnet_utils import conv_bn, conv_bn_relu

__all__ = ['repvgg8']


class VGG(nn.Module):

    def __init__(self, cfg, num_classes=100):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], 3)
        self.block1 = self._make_layers(cfg[1], cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    # def get_feat_modules(self):
    #     feat_m = nn.ModuleList([])
    #     feat_m.append(self.block0)
    #     feat_m.append(self.pool0)
    #     feat_m.append(self.block1)
    #     feat_m.append(self.pool1)
    #     feat_m.append(self.block2)
    #     feat_m.append(self.pool2)
    #     feat_m.append(self.block3)
    #     feat_m.append(self.pool3)
    #     feat_m.append(self.block4)
    #     feat_m.append(self.pool4)
    #     return feat_m

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        feat_m.append(self.classifier)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):  # is_feat表示是否返回中间的feature,preact表示是否返回激活之前的feature
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv_bn(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def repvgg8(**kwargs):
    """VGG 8-layer model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG([[64], [128], [256], [512], [512]], **kwargs)
    return model
