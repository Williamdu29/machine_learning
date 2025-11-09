import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载MNIST
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 只保留 0 和 1 的数据
train_indices = [i for i, (x, y) in enumerate(train_dataset) if y in [0, 1]]
test_indices = [i for i, (x, y) in enumerate(test_dataset) if y in [0, 1]]

train_dataset = Subset(train_dataset, train_indices) # Subset函数的作用是从原始数据集中抽取指定索引的数据
test_dataset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义三层神经网络
class ThreeLayerNN(nn.Module):
    def __init__(self):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 25)  # 输入层 → 25
        self.fc2 = nn.Linear(25, 15)     # 25 → 15
        self.fc3 = nn.Linear(15, 1)      # 15 → 1（输出）
        self.relu = nn.ReLU() # ReLU 激活函数

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出一个值
        return x

# 3. 初始化模型、损失函数、优化器
model = ThreeLayerNN()
criterion = nn.BCEWithLogitsLoss()  # 输出层没有sigmoid，这里自动处理
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练
epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader: # train_loader是一个可迭代对象，每次迭代返回一个batch的数据
        labels = labels.float().unsqueeze(1)  # [batch_size, 1]
        
        outputs = model(images) # outputs shape: [batch_size, 1]
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. 测试
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader: # test_loader是一个可迭代对象，每次迭代返回一个batch的数据
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        predicted = torch.sigmoid(outputs) > 0.5 # 阈值0.5，大于0.5为1，否则为0 
        # 二分类问题，模型的输出是一个概率值，表示样本属于正类（标签为1）的概率。为了将这个概率转换为具体的类别标签（0或1），我们通常会使用一个阈值（threshold）。最常用的阈值是0.5。
        correct += (predicted.int() == labels.int()).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
