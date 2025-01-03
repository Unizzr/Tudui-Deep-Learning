import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1,1,5,5)) #(batchsize, channels, h, w)
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True) #初始化使用的方法

    def forward(self, input):
        output = self.maxpool1(input) #调用方法
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



