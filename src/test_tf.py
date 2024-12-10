from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# python的用法 -> tensor数据类型
# 通过transforms.ToTensor去看两个问题

img_path = "../dataset/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

# 1、transforms该如何使用？（Python）
tensor_trans = transforms.ToTensor() #创建一个ToTensor类
tensor_img = tensor_trans(img) #传入所需参数获取来使用

# 2、为什么需要tensor数据类型？
writer.add_image("Tensor_img", tensor_img)

writer.close()