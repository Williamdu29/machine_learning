import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data
y = housing.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("训练集样本数目：", X_train.shape)
print("测试集样本数目：", X_test.shape)

lr = LinearRegression() #最小二乘法，不需要迭代训练
lr.fit(X_train, y_train)
print("回归系数：", lr.coef_)
print("回归截距：", lr.intercept_)

y_pred = lr.predict(X_test) # train的数据集用来训练权重，test的数据集用来测试模型
from sklearn import metrics
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print("MSE=", MSE)
print("RMSE=", RMSE)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = ["sans-serif"]
# 中文显示
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(15,5))
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',color='red')
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.show()