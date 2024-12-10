# 技巧：关注输入和输出类型，多看官方文档，关注方法需要什么参数
# 不知道返回值的时候，可以使用print，print(type())，debug来测试
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

writer = SummaryWriter("../logs")
img = Image.open("../image/kitty.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #参数为每个channel的均值和标准差
img_norm = trans_norm(img_tensor) #output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512)) #按照给定长宽裁剪
# img PIL --resize--> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL --totensor--> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(256) #长宽等比缩放
# PIL -> PIL ->tensor，将不通trans结合在一起使用
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()