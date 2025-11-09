# =============================
# 样本不平衡下逻辑回归正则化比较实验
# 数据集：Credit Card Fraud Detection
# =============================

'''
样本数量：284,807 条交易记录
时间跨度：2 天（欧洲信用卡用户的交易数据）
特征数量：30 个特征（columns）
其中 28 个经过 PCA 降维（V1 ~ V28）
Time：表示该笔交易与第一笔交易之间的时间差（以秒计）
Amount：交易金额
Class：标签字段
0 表示正常交易
1 表示欺诈交易（正样本）

类别	         数量	        占比
正常交易 (0)	 284,315	    99.83%
欺诈交易 (1)	 492	        0.17%
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# =============================
# 1. 加载数据集
# =============================
data = pd.read_csv("/Users/duchengtai/Library/Mobile Documents/com~apple~CloudDocs/GitHub repo/machine_learning/machine_learning_course_xjtu/project3/creditcard.csv")
X = data.drop("Class", axis=1) # 把 Class 列从数据中移除，得到特征矩阵 X
y = data["Class"] # 标签向量 y

print("原始样本分布：")
print(y.value_counts()) # 打印原始类分布，显示不平衡情况（正常样本极多，欺诈样本极少）

# =============================
# 2. 数据划分 + 标准化
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # 保证训练集和测试集中类别比例与原始数据一致（非常重要于不平衡问题）
)

scaler = StandardScaler() # 实例化标准化器
X_train = scaler.fit_transform(X_train) # 这一步对训练集进行 fit_transform，fit 是计算均值和标准差，transform 是应用标准化
X_test = scaler.transform(X_test) # 不要对测试集 fit，那会造成信息泄露

# =============================
# 3. 处理样本不平衡（SMOTE）
# =============================
smote = SMOTE(random_state=42) # 过采样器实例化
X_resampled, y_resampled = smote.fit_resample(X_train, y_train) # 在训练集上应用 SMOTE：为少数类（欺诈）生成合成样本，使训练集趋于平衡
print("\nSMOTE后样本分布：")
print(pd.Series(y_resampled).value_counts()) 

'''
SMOTE 只应用在训练集（不能对测试集进行过采样）。
SMOTE 生成的是合成样本，可能引入噪声或使模型过拟合，需谨慎并结合交叉验证观察性能。
'''

# =============================
# 4. 模型定义（L1, L2, ElasticNet）
# =============================
models = {
    "L1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000),
    "L2": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000),
    "ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1000) # "ElasticNet"：组合 L1 与 L2（penalty="elasticnet"），需要使用 solver="saga" 并指定 l1_ratio（L1 与 L2 的混合比例，0.5 表示各占一半）
}

# =============================
# 5. 训练与评估
# =============================
results = []

for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n===== {name} Regularization =====")
    print("AUC: {:.4f}".format(auc))
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision, recall, f1))
    print(classification_report(y_test, y_pred))
    
    results.append([name, auc, precision, recall, f1])

# =============================
# 6. 额外对比：使用class_weight='balanced'
# =============================
balanced_model = LogisticRegression(
    penalty="l2", solver="lbfgs", class_weight="balanced",  # 对少数类自动增大权重，逻辑回归在训练时会把少数类的错分惩罚提高，从而使模型更重视少数类样本，惩罚是根据类频率自动计算的
    max_iter=1000
)
balanced_model.fit(X_train, y_train)
# 这里没有使用 SMOTE，而是在原始训练集（已标准化但未过采样的）上训练。
# 这样可以对比两种思路：过采样（SMOTE）+ 普通训练 vs 不采样 + 类权重调整
y_pred_bal = balanced_model.predict(X_test)
y_prob_bal = balanced_model.predict_proba(X_test)[:, 1]

auc_bal = roc_auc_score(y_test, y_prob_bal)
f1_bal = f1_score(y_test, y_pred_bal)
precision_bal = precision_score(y_test, y_pred_bal)
recall_bal = recall_score(y_test, y_pred_bal)

print("\n===== L2 + class_weight=balanced =====")
print("AUC: {:.4f}".format(auc_bal))
print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_bal, recall_bal, f1_bal))

results.append(["L2 + balanced", auc_bal, precision_bal, recall_bal, f1_bal])


'''
SMOTE 通过增加少数类样本来让模型看到更多少数类样本，从数据层面缓解不平衡；
class_weight 通过修改损失函数对少数类加权，从算法层面缓解不平衡；
两者优劣依数据集而异，通常都值得比较。
'''

# =============================
# 7. 结果整理与可视化
# =============================
results_df = pd.DataFrame(results, columns=["Model", "AUC", "Precision", "Recall", "F1"])
print("\n=== 综合对比结果 ===")
print(results_df)

# 绘图
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(results_df["Model"], results_df["AUC"], color='steelblue')
plt.title("AUC Comparison")
plt.ylabel("AUC Score")

plt.subplot(1,2,2)
plt.bar(results_df["Model"], results_df["F1"], color='orange')
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")

plt.tight_layout()
plt.show()
