import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets =torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1,1,1,3))
targets = torch.reshape(targets, (1,1,1,3))

loss = nn.L1Loss(reduction='sum')
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
print(x.shape)
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss() #交叉熵
result_cross = loss_cross(x, y)

print(result)
print(result_mse)
print(result_cross)