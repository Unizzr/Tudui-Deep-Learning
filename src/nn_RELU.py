# 非线性激活
import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
#
# input = torch.reshape(input, (-1,1,2,2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.sigmoid1 = Sigmoid()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
