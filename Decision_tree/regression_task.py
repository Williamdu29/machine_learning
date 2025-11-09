import numpy as np
import matplotlib.pyplot as plt

# 构建数据
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5)**2
y = y + np.random.randn(m,1) / 10 # 加入一些噪声

# 可视化数据
plt.plot(X, y, 'go')
plt.show()


# 创建回归数对象并进行训练
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(random_state=41, max_depth=2) # max_depth是树的最大深度
tree_reg2 = DecisionTreeRegressor(random_state=41, max_depth=3)

tree_reg1.fit(X, y)
tree_reg2.fit(X, y)
# 在回归树中，判断分支好坏的指标是MSE（均方差），分支后的子节点均方差越小越好，代表分支后，子节点的样本都比较接近，分类效果较好

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1) # 形状(500,1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$") # 在x1上预测的结果

plt.figure(figsize=(11, 4))
plt.subplot(121)    
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k-"), (0.7718, "k-")): #在当前子图上画多条竖直分割线（常用于展示回归树分割点）
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)

plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2,X,y,ylabel=None)
for split,style in ((0.1973,"k-"),(0.0917,"k-"),(0.7718,"k-")):
    plt.plot([split,split],[-0.2,1],style,linewidth=2)
for split in (0.0458,0.1298,0.2873,0.9040):
    plt.plot([split,split],[-0.2,1],"k:",linewidth=1)
plt.text(0.3,0.5,"Depth=2",fontsize=13)
plt.title("max_depth=3",fontsize=14)
plt.show ()


# max_depth设置为3的模型较为复杂，max_depth设置为2的模型较为简单


# 如果不限制树的最大深度，决策树会一直划分下去，直到每个叶子节点只有一个样本为止，这样会导致过拟合现象
tree_reg3 = DecisionTreeRegressor(random_state=42)
tree_reg4 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg3.fit(X, y)
tree_reg4.fit(X, y)
x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred1 = tree_reg3.predict(x1)
y_pred2 = tree_reg4.predict(x1)
plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)
plt.legend(loc="upper center", fontsize=18)
plt.title("No restrictions", fontsize=14)

plt.subplot(122)
plt.plot(X,y,"b.")
plt.plot(x1,y_pred2,"r.-",linewidth=2,label=r"$\hat{y}$")
plt.axis([0,1,-0.2,1.1])
plt.xlabel("$x_1$",fontsize=18)
plt.title("min_samples_leaf=10".format (tree_reg2.min_samples_leaf),fontsize=14)
plt.show()
