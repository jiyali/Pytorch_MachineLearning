import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 设备配置
torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


# 训练数据集
train_dataset = torchvision.datasets.MNIST(root='../../../data/minist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

# 测试数据集
test_dataset = torchvision.datasets.MNIST(root='../../../data/minist',
                                          train=False,
                                          transform=transforms.ToTensor())

# 数据加载器
# 训练数据 加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 搭建卷积神经网络模型
# 两个卷积层
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            #  批归一化
            nn.BatchNorm2d(16),
            # ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    # 定义前向传播顺序
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# 实例化一个模型，并迁移至gpu
model = ConvNet(num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 注意模型在GPU中，数据也要搬到GPU中
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 切换成评估测试模式
# 这是因为在测试时，与训练时的dropout和batch normalization的操作是不同的
model.eval()

# 节省计算资源，不去计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    
#  保存模型
torch.save(model.state_dict(), 'model.ckpt')

