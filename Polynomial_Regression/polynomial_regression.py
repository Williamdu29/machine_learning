from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.uniform(-4,2,size=(100))
y = 2 * x**2 + 4 * x + 3 + np.random.randn(100)
x = x.reshape(-1,1) 
plt.scatter(x, y)

X_new = np.hstack((x**2, x)) # 多项式回归，增加x的高次项
linear_regression_new = LinearRegression()
linear_regression_new.fit(X_new, y)

y_predict_new = linear_regression_new.predict(X_new)
plt.scatter(x, y)
plt.plot(np.sort(x, axis=0), y_predict_new[np.argsort(x, axis=0)], color='red')

'''
x 的形状是 (n, 1)，表示 x 是一个二维数组（n 行 1 列）。
axis=0 表示沿着第 0 轴，也就是沿着行的方向排序。
因为 x 是一列数据，axis=0 确保对每一列（在这里就是 x 的唯一一列）中的值按行排序，这样可以将 x 从小到大排列。
'''

plt.show()

print("截距c=", linear_regression_new.intercept_)
print("回归系数a,b=", linear_regression_new.coef_) # coef_[