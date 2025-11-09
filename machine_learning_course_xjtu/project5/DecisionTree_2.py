# ===============================
# 📊 实际数据集 + 不平衡分类 + 模型比较
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# 1️⃣ 载入（乳腺癌数据集）

# 30个特征
# 569个样本
# 2个类别（0：良性，1：恶性）
data_raw = load_breast_cancer(as_frame=True) # as_frame=True 返回 DataFrame 格式
X = data_raw.data
y = data_raw.target

feature_names = X.columns
print("数据集形状:", X.shape)
print("类别分布:", np.bincount(y))
print(f"类别比例: {np.bincount(y)[1] / len(y):.3f} 为正样本")

# 2️⃣ 颜色映射 + 可视化部分（抽取部分特征看分布）
# 颜色映射：0 - 黄色，1 - 红色
color_map = {0: "yellow", 1: "red"}
color_list = [color_map[val] for val in y]

fig, axs = plt.subplots(3, 4, figsize=(18, 10), dpi=80)
axs = axs.flatten()
x_coord = np.linspace(0, len(y)-1, len(y))

for i, col in enumerate(feature_names[:len(axs)]):
    axs[i].scatter(x_coord, X[col], color=color_list, s=5)
    axs[i].set_title(col)
    axs[i].set_xlabel("Sample Index")
    axs[i].set_ylabel(col)

plt.tight_layout()
plt.show()

# 3️⃣ 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集正样本比例: {y_train.mean():.3f}")

# 4️⃣ 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ SMOTE 平衡
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"SMOTE 后正样本比例: {y_train_res.mean():.3f}")

# 6️⃣ 定义模型
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 7️⃣ 模型训练与评估
results = []
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(classification_report(y_test, y_pred))
    
    results.append((name, auc, precision, recall))
    
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# 8️⃣ 模型指标可视化
model_names = [r[0] for r in results]
auc_scores = [r[1] for r in results]
precisions = [r[2] for r in results]
recalls = [r[3] for r in results]

plt.figure(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.25

plt.bar(x - width, auc_scores, width=width, label="AUC")
plt.bar(x, precisions, width=width, label="Precision")
plt.bar(x + width, recalls, width=width, label="Recall")
plt.xticks(x, model_names)
plt.title("Model Performance Comparison（AUC / Precision / Recall）")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()
