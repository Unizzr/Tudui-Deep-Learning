import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Yujie(nn.Module):
    def __init__(self):
        super(Yujie, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

yujie = Yujie()

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, target = data
    output = yujie(imgs)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) 输出有6个channel，而彩色图像是3个channel
    output = torch.reshape(output, (-1, 3, 30, 30)) #第一个batchsize不知道写多少就写-1，会根据后面的数字来修改
    writer.add_images("output", output, step)

    step = step + 1
