import random as rd
from math import sqrt
import matplotlib.pyplot as plt

'''
num = 500  # 投掷点数为1000的整数倍
for t in range(1, 4+1):  # 分别观察1000、2000、3000、4000个投掷点的结果
    plt.subplot(2, 2, t)
    area = 0    # 落在1/4单位圆内的点数
    for i in range(0, num*t):
        x = rd.random()      # 随机生成投掷点的横坐标
        y = rd.random()
        r = sqrt(x**2 + y**2)   # 计算当前点到坐标原点距离
        if r <= 1:              # 判断当前点是否在单位圆内
            area += 1           # 在单位圆内，计数+1
            plt.scatter(x, y, color='blue', marker='.')   # 画散点图，单位圆内画蓝色
        else:
            plt.scatter(x, y, color='red', marker='.')    # 单位圆外画红色
    plt.xlabel('x')      # 坐标横轴标签
    plt.ylabel('y')
    plt.title("{} points Monte Carlo".format(t*num))    # 图名
    pi = area*4/(t*num)  # 根据蒙特卡罗法原理计算圆周率
    print('pi = %f' % pi)
plt.tight_layout()
plt.show()
'''

import numpy as np
np.random.seed(42)

num = 1000

plt.figure(figsize=(10, 10)) # 创建画布
for t in range(1,5):
    total_points = t * num # 投掷点数

    # 用numpy更加高效生成所有的投掷点，uniform是均匀分布，0-1区间内生成total_points个点
    x = np.random.uniform(0, 1, total_points)
    y = np.random.uniform(0, 1, total_points)

    # 计算每个点到原点的距离
    distances = np.sqrt(x**2 + y**2)

    # 判断每个点是否在单位圆内
    is_inside = distances <= 1 # inside:布尔数组

    # 计算落在1/4单位圆内的点数
    points_inside = np.sum(is_inside)

    # 计算圆周率
    pi = (points_inside / total_points) * 4

    plt.subplot(2, 2, t) # 创建子图

    x_inside = x[is_inside] # x,y分别取出在单位圆内和外的点
    y_inside = y[is_inside]
    x_outside = x[~is_inside]
    y_outside = y[~is_inside]

    plt.scatter(x_inside, y_inside, color='blue', marker='.', s=25) # 单位圆内的点 size=1
    plt.scatter(x_outside, y_outside, color='red', marker='.', s=25) # 单位圆外的点

    #绘制边界
    theta = np.linspace(0, np.pi/2, 100) # 生成0到π/2之间的100个点
    circle_x = np.cos(theta) # 计算对应的x坐标
    circle_y = np.sin(theta) # 计算对应的y坐标
    plt.plot(circle_x, circle_y, color='black', linewidth=1.5, linestyle='--') # 绘制边界线

    plt.xlim(0, 1) # 设置坐标轴范围
    plt.ylim(0, 1) 

    plt.xlabel('x') # 坐标横轴标签
    plt.ylabel('y')
    plt.title(f"{total_points} points Monte Carlo, pi={pi:.5f}") # 图名

    print(f'投掷点数: {total_points}, pi = {pi:.5f}')

plt.tight_layout() # 自动调整子图间距
plt.show() # 显示图像