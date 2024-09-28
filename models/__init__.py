from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, resnet50
from .resnetv2 import ResNet50
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .repvgg import repvgg8
from .represnet import represnet20, represnet32, represnet8x4
from .repmobilenetv2 import repmobilenetv2
from .repShuffleNetv1 import repShuffleV1
from .wrn import wrn_40_2
from .convnet_utils import switch_deploy_flag, switch_conv_bn_impl

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'resnet50': resnet50,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'repvgg8': repvgg8,
    'represnet8x4': represnet8x4,
    'represnet20': represnet20,
    'represnet32': represnet32,
    'repmobilenetv2': repmobilenetv2,
    'repshufflev1': repShuffleV1,
    'wrn_40_2': wrn_40_2
}
