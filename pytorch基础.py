import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# ############## 基础案例1 ###########

# # 创建张量
# x = torch.tensor(1., requires_grad=True)
# w = torch.tensor(2., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)
#
# # 构建计算图：前向计算
# y = w * x + b
#
# # 反向传播，计算梯度
# y.backward()
#
# # 输出梯度
# print(x.grad)
# print(w.grad)
# print(b.grad)

# ################基础案例2################

# 创建大小为(10,3)和(10,2)的张量
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 创建全连接层
linear = nn.Linear(3, 2)
print('w:', linear.weight)
print('b:', linear.bias)

# 构建损失函数（均方误差）和优化器（随机梯度下降）
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 前向传播
pred = linear(x)

# 计算损失
loss = criterion(pred, y)
print('loss', loss.item())

# 反向传播
loss.backward()

# 输出梯度
print('dL/dw:', linear.weight.grad)
print('dL/db:', linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())
