# 网络模型的读取
import torch
import torchvision
#from model_save import * #解决陷阱

# 方式1-->保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱 会报错，没有这个类，需要放入模型定义
model = torch.load("tudui_method1.pth")
print(model)