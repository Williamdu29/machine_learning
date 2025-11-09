## 最小二乘法一定是全局最小，但计算繁琐，且复杂情况下未必有解；
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[32,31],[53,68],[61,62],[47,71],[59,87],[55,78],[52,79],[39,59],[48,75],[52,71],
          [45,55],[54,82],[44,62],[58,75],[56,81],[48,60],[44,82],[60,97],[45, 48],[38,56],
          [66,83],[65,118],[47,57],[41,51],[51,75],[59,74],[57,95],[63,95],[46,79],[50,83]])

x = data[:,0]
y = data[:,1]

def compute_cost(w,b,points):
    total_cost = 0
    M = len(points)

    for i in range(M):
        x = points[i,0]
        y = points[i,1]
        total_cost += (y - w*x - b)**2  # 计算每个点的误差平方和
    
    return total_cost / M  # 返回均方误差

def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num

def fit(points):
    M = len(points)
    x_mean = average(points[:,0])
    
    sum_yx = 0 # Σ(xi- x_mean)(yi - y_mean)
    sum_xx = 0 # Σ(xi - x_mean)^2
    sum_delta = 0

    for i in range(M):
        x = points[i,0]
        y = points[i,1]
        sum_yx += (x - x_mean) * y
        sum_xx += x**2
    w = sum_yx / (sum_xx - M * x_mean**2) # 计算w

    for i in range(M):
        x = points[i,0]
        y = points[i,1]
        sum_delta += (y - w*x)
    b = sum_delta / M # 计算b
    return w, b

w, b = fit(data)

print("w=", w, "b=", b)
print("cost=", compute_cost(w, b, data))

plt.scatter(x, y)

pred_y = w * x + b
plt.plot(x, pred_y, color='red')
plt.show()