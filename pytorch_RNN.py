# 包
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 设备配置
# Device configuration
torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # 选用LSTM RNN结构
        self.fc = nn.Linear(hidden_size, num_classes)  # 最后一层为全连接层，将隐状态转为分类

    def forward(self, x):
        # 初始化隐层状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # 输出大小 (batch_size, seq_length, hidden_size)

        # 解码最后一个时刻的隐状态
        out = self.fc(out[:, -1, :])
        return out

# 实例化一个模型
# 注意输入维度，虽然我不懂将一幅图28x28拆成28个大小为28的序列有啥意义
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
# Adam: A Method for Stochastic Optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)  # 注意维度
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化，注意梯度每次清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 测试集
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), 'model.ckpt')
