import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 超参数设置
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../../data/minist',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../../../data/minist',
                                          train=False,
                                          transform=transforms.ToTensor())


# 数据加载器（data loader）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 线性模型，指定
model = nn.Linear(input_size, num_classes)

# 损失函数和优化器
# nn.CrossEntropyLoss()内部集成了softmax函数
# It is useful when training a classification problem with `C` classes.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像序列抓换至大小为 (batch_size, input_size)
        images = images.reshape(-1, 28 * 28)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播及优化
        optimizer.zero_grad()  # 注意每次循环都要注意清空梯度缓存
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
