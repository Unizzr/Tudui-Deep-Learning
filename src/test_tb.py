# 按ctrl键点击函数或库查看作用
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 使用命令 tensorboard --logdir=logs --port=6007 打开生成的文件
writer = SummaryWriter("../logs")
img_path = "../dataset/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(img_path) #获取图片文件
img_array = np.array(img_PIL) #将图片数据转为numpy类型，原因见add_image函数说明

print(type(img_array)) #查看数据类型，成功转换为numpy格式
print(img_array.shape) #查看数据形状 (512, 768, 3),为HWC格式，见函数说明
writer.add_image("test", img_array, 3, dataformats="HWC")

# y = x
for i in range(100):
    writer.add_scalar("y=x", i, i) #写入标量


writer.close()