# 预剪枝参数的介绍
# min_samples_split：节点在分割之前必须具有的最小样本数
# min_samples_.leaf：叶子节点必须具有的最小样本数
# max_leaf_nodes：叶子节点的最大数量
# max_features：在每个节点处评估用于拆分的最大特征数（除非特征非常多，否则不建议限制最大特征数）
# max_depth：树最大的深度

# 预剪枝的作用：缓解模型过拟合
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import numpy as np
import matplotlib.pyplot as plt

# 定义绘制决策边界的函数
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    # axes是x1和x2轴的范围，iris是是否是鸢尾花数据集，plot_training是是否绘制训练数据点
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()] # 把网格展平并按列拼成 (num_points, 2) 的矩阵，传给分类器进行预测
     # ravel()将多维数组展平为一维数组，c_[]按列拼接
     # 例如：x1 = [[1,2,3],[4,5,6]], x2 = [[7,8,9],[10,11,12]]
     # 则X_new = [[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]]
     # shape = (6, 2)
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap) 
    if not iris:
        # 这里是非鸢尾花数据集的情况
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8) 

    if plot_training: # 绘制训练数据点
        # 对每个类别用固定的标记和标签绘制
        # 这里硬编码了3个类别
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-virginica")
        plt.axis(axes)
     
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=53)
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42) # 预剪枝

tree_clf1.fit(X, y)
tree_clf2.fit(X, y)

plt.figure(figsize=(12,4))
plt.subplot(121)
plot_decision_boundary(tree_clf1, X, y, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)

plt.subplot(122)
plot_decision_boundary(tree_clf2, X, y, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = 4", fontsize=16)
plt.show()

#左边是没有增加预剪枝参数的决策边界，明显可以看出，它将一些离群点也考虑进去了，模型过为复杂，存在过拟合现象
#而右边限制了 min_samples_leaf = 4 的决策树就没有存在明显的过拟合现象。


