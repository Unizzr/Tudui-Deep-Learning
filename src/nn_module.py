import torch
from torch import nn

# 神经网络模型
class YuJie(nn.Module):
    def __init__(self, *args, **kwargs) -> None: #可以使用代码重写补全
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output

yujie = YuJie()
x = torch.tensor(1.0)
output = yujie(x)
print(output)