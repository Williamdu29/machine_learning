import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) # 设置随机种子，保证每次生成的随机数相同，便于调试

# 生成随机数据
# 2个特征的均值和方差
mean_1 = [2, 2]
cov_1 = [[2, 0], [0, 2]]  # 对角协方差矩阵，表示两个特征之间不相关
mean_2 = [-2, -2]
cov_2 = [[1, 0], [0, 1]]

# 生成类别1的数据
x1 = np.random.multivariate_normal(mean_1, cov_1, 50)
y1 = np.zeros(50)  # 类别1的标签为0

# 生成类别2的数据
x2 = np.random.multivariate_normal(mean_2, cov_2, 50)
y2 = np.ones(50)  # 类别2的标签为1

# 合并数据
X = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2))

# 可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data for Logistic Regression')
plt.show()

def sigmoid(z):
    if z>0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))
    # return 1 / (1 + np.exp(-z)) # 可能会溢出

class Logistic_Regression:
    def __init__(self, learning_rate=0.01, num_iter=1000):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 初始化权重和偏置
        self.w = np.zeros(num_features)
        self.b = 0

        for i in range(self.num_iter):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = np.array([sigmoid(z) for z in linear_model])

            # 计算梯度
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # 更新权重和偏置
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = np.array([sigmoid(z) for z in linear_model])
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    

logreg = Logistic_Regression()
logreg.fit(X, y)

X_new = np.array([[2.5, 2.5],[-6.0, -4.0]])
y_pred = logreg.predict(X_new)
print("Predictions for new samples:", y_pred)

    


