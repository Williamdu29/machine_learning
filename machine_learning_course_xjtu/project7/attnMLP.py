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

# ---------- 数据加载（与你原来相同） ----------
data = pd.read_csv("machine_learning_course_xjtu/project7/creditcard.csv")
X = data.drop(columns=['Class'])
y = data['Class'].values

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X = X.drop(columns=['Time'])

X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 转为 tensor
X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

input_dim = X_train.shape[1]  # 应为 29

# ---------- 改进模型：Feature Attention + MLP + Residual ----------
class FeatureAttention(nn.Module):
    """对输入每个特征计算注意力权重（per-feature），并返回加权特征"""
    def __init__(self, dim, hidden=64):
        super().__init__()
        # 小型 MLP 输出每个特征的得分（可用 softmax 或 sigmoid）
        # 这里用两层 MLP，输出 dim 分数，再用 sigmoid 缩放到 (0,1)
        self.net = nn.Sequential( # self.net 是一个小型的前馈神经网络（MLP），它的任务是 根据输入 x 学习到一组特征权重
            nn.Linear(dim, hidden), # dim是输入特征数
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (batch, dim)
        w = self.net(x)         # (batch, dim) — 每样本的特征权重
        return x * w, w        # 返回加权后的特征与权重，重要的特征权重大，不重要的特征权重小

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], dropout=0.0):
        super().__init__()
        self.att = FeatureAttention(input_dim, hidden=64)

        layers = []
        prev = input_dim
        self.bns = nn.ModuleList()
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            # no dropout as per your best param (dropout=0.0)
            self.bns.append(nn.BatchNorm1d(h)) # batchnorm after each hidden layer
            prev = h
        self.mlp = nn.Sequential(*layers)
        # final layer
        self.fc_out = nn.Linear(prev, 1)
        self.sig = nn.Sigmoid()

        # if input and first hidden have same dim, we can do residual; here they differ, but we'll add a simple skip:
        if input_dim == hidden_dims[0]:
            self.use_residual = True
        else:
            self.use_residual = False

    def forward(self, x):
        # x: (batch, input_dim)
        x_att, att_weights = self.att(x)   # (batch, input_dim), (batch, input_dim)
        h = x_att
        residual = h  # for potential residual connection

        idx = 0
        # pass through pairs of (Linear, ReLU) and BatchNorm
        for layer in self.mlp:
            h = layer(h)
            # apply batchnorm after each linear+relu pair by tracking index
            if isinstance(layer, nn.ReLU):
                # corresponding bn index
                bn = self.bns[idx]
                h = bn(h)
                idx += 1
                
        # optional residual connection
        if self.use_residual and residual.shape == h.shape:
            h = h + residual

        out = self.fc_out(h)   # (batch,1)
        out = self.sig(out).squeeze(-1)  # (batch,)
        return out, att_weights  # 返回预测概率与注意力权重

# ---------- 损失 / 训练函数（使用你指定的超参：Adam, lr=1e-3, hidden=[128,64], dropout=0） ----------
def train_eval_attention(hidden_dims=[128,64], lr=1e-3, batch_size=512, epochs=10, weight_pos=None, use_focal=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionMLP(input_dim, hidden_dims=hidden_dims, dropout=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 可以选择 class weight 或不设（我们先不设）
    criterion = nn.BCELoss()

    Xtr = X_train.to(device); ytr = y_train.to(device)
    Xte = X_test.to(device); yte = y_test.to(device)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(Xtr.size(0))
        epoch_loss = 0.0
        for i in range(0, Xtr.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        # tqdm or print
        # print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss / Xtr.size(0):.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        y_prob, attw = model(Xte)
        y_prob = y_prob.cpu().numpy()
        attw = attw.cpu().numpy()  # shape (n_test, input_dim)
    auc = roc_auc_score(y_test_np, y_prob)
    prauc = average_precision_score(y_test_np, y_prob)
    f1 = f1_score(y_test_np, (y_prob>0.5).astype(int))
    return {'model': model, 'auc': auc, 'prauc': prauc, 'f1': f1, 'att_weights': attw, 'y_prob': y_prob}

# ---------- 执行一次训练评估 ----------
res = train_eval_attention(hidden_dims=[128,64], lr=1e-3, batch_size=512, epochs=10)
print("AUC:", res['auc'], "PR-AUC:", res['prauc'], "F1:", res['f1'])

# ---------- 可视化：展示特征注意力均值（哪些特征被模型关注） ----------
import numpy as np
mean_att = res['att_weights'].mean(axis=0)  # (input_dim,)
feat_names = X_train_df.columns.tolist()
plt.figure(figsize=(10,4))
plt.bar(range(len(feat_names)), mean_att)
plt.xticks(range(len(feat_names)), feat_names, rotation=45, ha='right')
plt.ylabel("Mean attention weight")
plt.title("Feature attention (mean over test samples)")
plt.tight_layout()
plt.show()
