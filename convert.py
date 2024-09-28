import argparse
import os
import torch
from models.convnet_utils import switch_conv_bn_impl, switch_deploy_flag
from models import model_dict

parser = argparse.ArgumentParser(description='DBB Conversion')
parser.add_argument('--load', metavar='LOAD', help='path to the weights file',
                    default='save/student_model/actdb/vgg8_vgg13_actdb/S_repvgg8_T_vgg13_cifar100_repkd_r_0.1_a_0.9_b_0_actdb_5/repvgg8_last.pth')
parser.add_argument('--save', metavar='SAVE', help='path to the weights file',
                    default='save/student_model/actdb/vgg8_vgg13_actdb/S_repvgg8_T_vgg13_cifar100_repkd_r_0.1_a_0.9_b_0_actdb_5/vgg13->vgg8_deploy_74.91.pth')
parser.add_argument('-a', '--arch', metavar='ARCH', default='repvgg8',
                    choices=['resnet8', 'resnet14', 'resnet20', 'represnet20', 'resnet32', 'resnet44', 'ResNet50',
                                 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'repvgg8', 'vgg8', 'vgg11',
                                 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2', 'repmobilenetv2'])
parser.add_argument('-t', '--blocktype', metavar='BLK', default='AMBB', choices=['AMBB'])


def convert():
    args = parser.parse_args()

    switch_conv_bn_impl(args.blocktype)
    switch_deploy_flag(False)
    train_model = model_dict[args.arch]()

    if 'hdf5' in args.load:
        from models.util import model_load_hdf5
        model_load_hdf5(train_model, args.load)
    elif os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    for m in train_model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    torch.save(train_model.state_dict(), args.save)


if __name__ == '__main__':
    convert()
