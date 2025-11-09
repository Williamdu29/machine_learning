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

alpha = 0.0001
init_w = 0
init_b = 0
num_iter = 10000

def gradient_descent(data, init_w, init_b, alpha, num_iter):
    w = init_w
    b = init_b

    cost_list = []
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, data))
        # 更新 w, b
        w,b = step_gradient(w,b,alpha,data)
    return w, b, cost_list

def step_gradient(cur_w, cur_b, alpha, data):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(data)

    for i in range(M):
        x = data[i, 0]
        y = data[i, 1]
        sum_grad_w += (cur_w * x + cur_b - y) * x
        sum_grad_b += (cur_w * x + cur_b - y)
    
    grad_w = (2/M) * sum_grad_w
    grad_b = (2/M) * sum_grad_b

    new_w = cur_w - grad_w * alpha
    new_b = cur_b - grad_b * alpha

    return new_w, new_b

w, b , cost_list = gradient_descent(data, init_w, init_b, alpha, num_iter)
print("w=", w, "b=", b)
# plt.plot(cost_list)
# plt.show()


cost = compute_cost(w, b, data)
print("cost=", cost)

plt.scatter(x, y)
pred_y = w * x + b
plt.plot(x, pred_y, color='red')
plt.show()
