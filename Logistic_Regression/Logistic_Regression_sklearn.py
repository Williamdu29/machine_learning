import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 训练集
# 设置随机种子，以便结果可复现
np.random.seed(42)

# 生成随机数据
# 两个特征的均值和方差
mean_1 = [2, 2]
cov_1 = [[2, 0], [0, 2]]
mean_2 = [-2, -2]
cov_2 = [[1, 0], [0, 1]]

# 生成类别1的样本
X1 = np.random.multivariate_normal(mean_1, cov_1, 50)
y1 = np.zeros(50)

# 生成类别2的样本
X2 = np.random.multivariate_normal(mean_2, cov_2, 50)
y2 = np.ones(50)

# 合并样本和标签
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
