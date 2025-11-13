import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- 数据加载 ----------
data = pd.read_csv("machine_learning_course_xjtu/project7/creditcard.csv")
X = data.drop(columns=['Class'])
y = data['Class'].values

scaler = StandardScaler() # 创建标准化器实例
X['Amount'] = scaler.fit_transform(X[['Amount']]) # 用整个数据的 Amount 列计算均值和标准差（fit），并把 Amount 列替换为标准化后的值（transform）
X = X.drop(columns=['Time']) # 删除 Time 列（通常是记录时间/顺序，对预测无用或会引入偏差），从而不作为模型输入

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 数据转换 ----------
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ---------- 模型定义 ----------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim # 输入层维度
        for h in hidden_dims: # 隐藏层维度列表 
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers) # 使用 Sequential 将层列表组合成网络

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze() # 输出层使用 Sigmoid 激活函数，适用于二分类，squeeze 去掉多余维度, 返回形状为 (N,)

# ---------- 训练函数 ----------
def train_eval(hidden_dims, lr, batch_size, dropout, optimizer_type, epochs=10):
    model = MLP(X_train.shape[1], hidden_dims, dropout) 
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCELoss() # 二分类交叉熵损失函数

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size] # 每次取一个批次
            yb = y_train[i:i+batch_size]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_prob = model(X_test).numpy()
    auc = roc_auc_score(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, (y_prob>0.5).astype(int))
    return auc, prauc, f1


# ---------- 参数搜索 ----------
params = []
for lr in [1e-4, 1e-3, 1e-2]: # 学习率
    for dropout in [0.0, 0.3, 0.5]: # Dropout 比例
        for hidden in [[64], [128,64], [256,128,64]]: # 隐藏层结构
            for opt in ['SGD', 'Adam']:
                auc, prauc, f1 = train_eval(hidden, lr, 512, dropout, opt)
                params.append({
                    'lr': lr,
                    'dropout': dropout,
                    'hidden': str(hidden),
                    'optimizer': opt,
                    'roc_auc': auc,
                    'pr_auc': prauc,
                    'f1': f1
                })
                print(params[-1])

df = pd.DataFrame(params)
print(df.sort_values("pr_auc", ascending=False).head())

# ---------- 可视化 ----------
plt.figure(figsize=(8,5))
for opt in ['SGD','Adam']:
    subset = df[df['optimizer']==opt]
    plt.plot(subset['lr'], subset['pr_auc'], 'o-', label=f"{opt}")
plt.xscale('log')
plt.xlabel("Learning rate")
plt.ylabel("PR AUC")
plt.legend()
plt.title("PR AUC vs Learning rate for different optimizers")
plt.show()
