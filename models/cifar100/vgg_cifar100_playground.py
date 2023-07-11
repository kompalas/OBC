'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
from distiller import project_dir
from collections import OrderedDict


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11_cifar100(pretrained=None):
    model = VGG('VGG11')
    if pretrained is not None:
        #chpt = os.path.join(os.path.dirname(project_dir),
        #                    'cifar10_100_playground', 'cifar100',
        #                    'models', 'vgg11_cifar100.pth')
        chpt = pretrained
        checkpoint = torch.load(chpt)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


def vgg13_cifar100(pretrained=None):
    model = VGG('VGG13')
    if pretrained is not None:
        #chpt = os.path.join(os.path.dirname(project_dir),
        #                    'cifar10_100_playground', 'cifar100',
        #                    'models', 'vgg13_cifar100.pth')
        chpt = pretrained
        checkpoint = torch.load(chpt)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


def vgg16_cifar100(pretrained=None):
    model = VGG('VGG16')
    if pretrained is not None:
        #chpt = os.path.join(os.path.dirname(project_dir),
        #                    'cifar10_100_playground', 'cifar100',
        #                    'models', 'vgg16_cifar100.pth')
        chpt = pretrained
        checkpoint = torch.load(chpt)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


def vgg19_cifar100(pretrained=None):
    model = VGG('VGG19')
    if pretrained is not None:
        #chpt = os.path.join(os.path.dirname(project_dir),
        #                    'cifar10_100_playground', 'cifar100',
        #                    'models', 'vgg19_cifar100.pth')
        chpt = pretrained
        checkpoint = torch.load(chpt)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
