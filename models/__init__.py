import copy
from functools import partial
import torch
import torchvision.models as torch_models
import torch.nn as nn
from . import cifar10 as cifar10_models
from . import cifar100 as cifar100_models
import pretrainedmodels

import logging
msglogger = logging.getLogger()

SUPPORTED_DATASETS = ('imagenet', 'cifar10', 'cifar100')

# ResNet special treatment: we have our own version of ResNet, so we need to over-ride
# TorchVision's version.
RESNET_SYMS = ('ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2')

TORCHVISION_MODEL_NAMES = sorted(
                            name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))

IMAGENET_MODEL_NAMES = copy.deepcopy(TORCHVISION_MODEL_NAMES)
IMAGENET_MODEL_NAMES.extend(pretrainedmodels.model_names)

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

CIFAR100_MODEL_NAMES = sorted(name for name in cifar100_models.__dict__
                              if name.islower() and not name.startswith("__")
                              and callable(cifar100_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES + CIFAR100_MODEL_NAMES)))


def patch_torchvision_mobilenet_v2(model):
    """
    Patches TorchVision's MobileNetV2:
    * To allow quantization, this adds modules for tensor operations (mean, element-wise addition) to the
      model instance and patches the forward functions accordingly
    * Fixes a bug in the torchvision implementation that prevents export to ONNX (and creation of SummaryGraph)
    """
    if not isinstance(model, torch_models.MobileNetV2):
        raise TypeError("Only MobileNetV2 is acceptable.")

    def patched_forward_mobilenet_v2(self, x):
        x = self.features(x)
        # x = x.mean([2, 3]) # this was a bug: https://github.com/pytorch/pytorch/issues/20516
        x = self.mean32(x)
        x = self.classifier(x)
        return x
    model.mean32 = nn.Sequential(
        Mean(3), Mean(2)
    )
    model.__class__.forward = patched_forward_mobilenet_v2

    def is_inverted_residual(module):
        return isinstance(module, nn.Module) and module.__class__.__name__ == 'InvertedResidual'

    def patched_forward_invertedresidual(self, x):
        if self.use_res_connect:
            return self.residual_eltwiseadd(self.conv(x), x)
        else:
            return self.conv(x)

    for n, m in model.named_modules():
        if is_inverted_residual(m):
            if m.use_res_connect:
                m.residual_eltwiseadd = EltwiseAdd()
            m.__class__.forward = patched_forward_invertedresidual


_model_extensions = {}


def create_model(arch, device_ids=None):
    """Create a pytorch model based on the model architecture
    """
    #return torch_models.resnet18(pretrained=True).to('cuda')
    dataset = 'imagenet'
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar' in arch:
        dataset = 'cifar10'

    model = None
    cadene = False
    if dataset == 'imagenet':
        model, cadene = _create_imagenet_model(arch)
    elif dataset == 'cifar10':
        model = _create_cifar10_model(arch)
    elif dataset == 'cifar100':
        model = _create_cifar100_model(arch)
    msglogger.info("=> created a {} model with the {} dataset".format(arch, dataset))
    if torch.cuda.is_available():
        device = 'cuda'
        #if arch.startswith('alexnet') or arch.startswith('vgg'):
        #    model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
        #else:
        #    model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    model.arch = arch
    model.dataset = dataset
    return model.to(device)


def is_inception(arch):
    return arch in [ # Torchvision architectures
                    'inception_v3', 'googlenet',
                    # Cadene architectures
                    'inceptionv3', 'inceptionv4', 'inceptionresnetv2']


def _create_imagenet_model(arch, pretrained=True):
    dataset = "imagenet"
    cadene = False
    model = None
    if arch in TORCHVISION_MODEL_NAMES:
        try:
            if is_inception(arch):
                model = getattr(torch_models, arch)(pretrained=pretrained, transform_input=False)
            else:
                model = getattr(torch_models, arch)(pretrained=pretrained)
            if arch == "mobilenet_v2":
                patch_torchvision_mobilenet_v2(model)

        except NotImplementedError:
            # In torchvision 0.3, trying to download a model that has no
            # pretrained image available will raise NotImplementedError
            if not pretrained:
                raise
    if model is None and (arch in pretrainedmodels.model_names):
        cadene = True
        model = pretrainedmodels.__dict__[arch](
            num_classes=1000,
            pretrained=(dataset if pretrained else None))
    if model is None:
        error_message = ''
        if arch not in IMAGENET_MODEL_NAMES:
            error_message = "Model {} is not supported for dataset ImageNet".format(arch)
        elif pretrained:
            error_message = "Model {} (ImageNet) does not have a pretrained model".format(arch)
        raise ValueError(error_message or 'Failed to find model {}'.format(arch))
    return model, cadene


def _create_cifar10_model(arch, pretrained=False):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model

def _create_cifar100_model(arch, pretrained=False):
    if pretrained:
        raise ValueError("Model {} (CIFAR100) does not have a pretrained model".format(arch))
    try:
        model = cifar100_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


