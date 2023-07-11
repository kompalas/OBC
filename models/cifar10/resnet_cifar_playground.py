'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from distiller.modules import EltwiseAdd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False) # added for pruning/quantization compatibility
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.residual_eltwise_add = EltwiseAdd() # added for pruning/quantization compatibility
        self.relu2 = nn.ReLU(inplace=False) # added for pruning/quantization compatibility

    def forward(self, x):
        # changed for pruning/quantization compatibility according to distiller rules
        res = out = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            res = self.shortcut(x)
        out = self.residual_eltwise_add(res, out)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False) # added for pruning/quantization compatibility
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False) # added for pruning/quantization compatibility
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.residual_eltwise_add = EltwiseAdd() # added for pruning/quantization compatibility
        self.relu3 = nn.ReLU(inplace=False) # added for pruning/quantization compatibility

    def forward(self, x):
        # modified for pruning/quantization compatibility according to distiller rules
        res = out = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.shortcut is not None:
            res = self.shortcut(x)
        out = self.residual_eltwise_add(res, out)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False) # added for pruning/quantization compatibility
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool2d = nn.AvgPool2d(4) # added for pruning/quantization compatibility
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x))) # changed for pruning/quantization compatibility
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out) # changed for pruning/quantization compatibility
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18_cifar(pretrained=None):
    model = ResNet(BasicBlock, [2,2,2,2])
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model



def resnet34_cifar(pretrained=None):
    model = ResNet(BasicBlock, [3,4,6,3])
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def resnet50_cifar(pretrained=None):
    model = ResNet(Bottleneck, [3,4,6,3])
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def resnet101_cifar(pretrained=None):
    model = ResNet(Bottleneck, [3,4,23,3])
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def resnet152_cifar(pretrained=None):
    model = ResNet(Bottleneck, [3,8,36,3])
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
