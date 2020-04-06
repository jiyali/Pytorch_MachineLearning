# 包
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# 设备配置
# 有cuda就用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# 定义：有一个隐藏层的全连接的神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 加载（实例化）一个网络模型
# to(device)可以用来将模型放在GPU上训练
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# 定义损失函数和优化器
# 再次，损失函数CrossEntropyLoss适合用于分类问题，因为它自带SoftMax功能
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将tensor移动到配置好的设备上（GPU）
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 还是要注意此处，每次迭代训练都需要清空梯度缓存
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 测试
# 测试阶段为提高效率，可以不计算梯度
# 使用with torch.no_grad()函数

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # 统计预测概率最大的下标
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


torch.save(model.state_dict(), 'model.ckpt')