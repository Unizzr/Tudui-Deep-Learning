from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 保存数据集的根目录路径，如dataset/train
        self.label_dir = label_dir  # 保存标签子目录的名称，如ants
        self.path = os.path.join(self.root_dir, self.label_dir)  # 拼接得到完整的标签目录路径dataset/train/ants
        self.img_path = os.listdir(self.path)  # 获取该目录下所有文件的列表（图像文件路径）

    def __getitem__(self, idx):
        img_name = self.img_path[idx]   #获取列表中的某个文件
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)   #获取该具体文件的路径
        img = Image.open(img_item_path)   #打开该文件
        label = self.label_dir  #获取文件标签
        return img, label   #返回该具体文件及其标签

    def __len__(self):
        return len(self.img_path)   #获取该标签下的文件数量


root_dir = "../dataset1/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset #通常用于真实数据集不足，需要自制数据集与其相结合使用时