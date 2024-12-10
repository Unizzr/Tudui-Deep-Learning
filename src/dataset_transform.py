import torchvision
from torch.utils.tensorboard import SummaryWriter

# 将数据集中的图片转换成tensor类型
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0]) #查看数据集中的数据类型，发现是【图片，target类别】，这一步即获取返回值类型
# print(test_set.classes)
#
# img, target = test_set[0] #接受返回值
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("../logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("", img, i)

writer.close()

