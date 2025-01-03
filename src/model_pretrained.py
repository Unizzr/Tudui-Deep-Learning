import torchvision
from torch import nn
from torchaudio.utils import download

# train_data = torchvision.datasets.ImageNet("../dataset", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT)
vgg16_true = torchvision.models.vgg16(torchvision.models.VGG16_Weights.IMAGENET1K_V1)

print(vgg16_true) #发现最后输出分为了1000个类

# CIFAR10只将图片分为了10类
train_data = torchvision.datasets.CIFAR10('../dataset', train=True, transform=torchvision.transforms.ToTensor())

# 添加一个层级
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 在已有层级上进行修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

