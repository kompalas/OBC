#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This package contains CIFAR image classification models for pytorch"""

from .simplenet_cifar import *
from .resnet_cifar import *
from .preresnet_cifar import *
#from .vgg_cifar import *
from .resnet_cifar_earlyexit import *
from .plain_cifar import *
# added for extra cifar10 models from cifar10_100_playground
from .resnet_cifar_playground import *
from .vgg_cifar_playground import *
from .efficientnet_cifar_playground import *
from .mobilenet_cifar_playground import *
from .mobilenetv2_cifar_playground import *
