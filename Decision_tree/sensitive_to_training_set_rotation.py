from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from pruning import plot_decision_boundary

np.random.seed(6)
Xs = np.random.rand(100, 2) - 0.5 # 生成100个二维点，范围在[-0.5, 0.5)
ys = (Xs[:, 0] > 0).astype(np.int32)*2 # ys 就是类别标签：如果点在 x>0 区域，标签为 2，否则为 0

angle = np.pi / 4  # 逆时针旋转45度
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

Xs_r = Xs.dot(rotation_matrix)  # 旋转数据集

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(Xs, ys)

tree_clf_r = DecisionTreeClassifier(random_state=42)
tree_clf_r.fit(Xs_r, ys)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(tree_clf, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.title("Original data set", fontsize=16)

plt.subplot(122)
plot_decision_boundary(tree_clf_r, Xs_r, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.title("Rotated data set", fontsize=16)
plt.show()


                                    
# 先看左图，决策树很轻松的用一根垂直线将样本分成了两份，但如果我们对数据做一点小小的改动，将原本的数据进行90度旋转，如右图所示，决策边界就会复杂很多。

# 主要原因：决策树进行决策边界划分时只能沿着与坐标轴垂直的方向划分，所以对数据很敏感