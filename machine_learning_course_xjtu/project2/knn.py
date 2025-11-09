from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix # 混淆矩阵，直观展示分类结果


def lookdataset(iris):
    """
    lookdataset is a function to look some basic information of the dataset.
    parameter : the dataset you load.
    returns : some "prints" you can see below, containing some shape information and 5 samples.

    """
    print('\nHere is the output of function lookdataset: \n')
    print(iris.keys())  # 观察数据集构成
    print('The shape of iris_target is :', iris.target.shape)  # 样本类别的维度 150*1
    print('The 3 types of flowers are :', iris.target_names)  # 三个类别的名称
    print('The shape of iris_data is :', iris.data.shape)  # 样本特征的维度 150*4
    print('The 4 features are :', iris.feature_names)  # 罗列4个特征
    print('some of features data :')
    print(iris.data[:5])  # 展示部分特征数据


def lookfeatures(iris):
    #哪些特征组合可以很好地区分类别
    #哪些特征之间有明显的相关性
    #哪些类别在特征空间中容易混淆

    """
    lookfeatures is a function to look the relationships between every two features.
    parameter : the dataset you load.
    returns : 16 plots of the single-single feature relationship;
              to help understand, here also provide 5 samples of data after merging the
              features and labels.
    """
    print('\nHere is the output of function lookdataset: \n')
    sns.set(style="white", color_codes=True)
    iris_frame = pd.DataFrame(iris['data'], columns=iris['feature_names']) # iris_data:(150,4)
    # 把鸢尾花特征数据结构变为dataframe，可该行下面加print观察
    iris_frame = pd.merge(iris_frame, pd.DataFrame(iris['target'], columns=['species']), left_index=True, right_index=True)    #把类别（iris['target']，即0/1/2）作为新的一列 'species' 加到表格最右边 
    # 在dataframe最右加多1列标签
    labels = dict(zip([0, 1, 2], iris['target_names']))     # 生成字典，建立0，1，2和三个类别的名字的映射
    iris_frame['species'] = iris_frame['species'].apply(lambda x: labels[x])  # 利用上述映射把标签列的内容（012）对应为类别名
    # lambda x: labels[x] 是一个匿名函数，输入x输出labels[x]
    print(iris_frame[:5])          # 查看设计好的dataframe
    sns.pairplot(iris_frame, hue='species', size=3, diag_kind='hist') #自动画出每对特征之间的二维散点图,diag_kind='hist' 对角线绘制直方图   
    # 绘制两两特征图，参数的数据结构为dataframe
    plt.show()

    # petal length vs petal width 最能区分三类花，花瓣长度和花瓣宽度几乎可以完美区分三种鸢尾花
    # sepal length vs sepal width 区分度不强
    # petal length vs sepal length 较好的线性相关性

    # sepal width 的直方图中，各类别分布相近，说明这个特征区分性较弱

def getparallel(data_origin):     # 获取平行坐标图，同样要先生成dataframe
    data = data_origin["data"] #(150,4)
    target = data_origin["target"] #类别编号
    target_names = data_origin["target_names"]  # 相当于查找表，与前面的字典作用相同
    target_labels = []
    for class_num in target:     # 利用查找表把标签列的内容（012）对应为类别名
        target_labels.append(target_names[class_num])

    feature_names = data_origin["feature_names"] # 特征名列表
    # 合成字典
    data_dict = {}   # 可以直接把一个字典变为dataframe的结构，所以把特征、标签等内容先合成字典
    column = 0
    for feature_name in feature_names:  # feature_name="sepal length", sepal width, .. , ..
        data_dict[feature_name] = data[:, column] # 选取所有行和对应的列 
        # dict={"sepal length":data[:,0] , ....    }
        column += 1
    data_dict["target_labels"] = target_labels #键为特征名，值为特征列
    # 合成dataFrame
    pd_data = pd.DataFrame(data_dict)    # 字典构建好后，直接转成dataframe结构，字典各栏目即为dataframe各列
    # 画图
    plt.figure()
    pd.plotting.parallel_coordinates(pd_data, "target_labels")     # 绘制平行坐标
    # 每个特征是一条纵轴；每个样本是一条线；每条线在各特征轴上取值对应；颜色按类别区分。
    plt.show()


iris = datasets.load_iris()  # 读入鸢尾花数据集
lookdataset(iris)  # 获得数据集相关信息
lookfeatures(iris)  # 查看特征两两之间分布情况

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=0)
print('The shape of x_train is :', x_train.shape)  # 100个训练样本，每个样本4个特征，1个标签
print('The shape of y_train is :', y_train.shape)
print('The shape of x_test is :', x_test.shape)    # 50个测试样本，每个样本4个特征，1个标签
print('The shape of y_test is :', y_test.shape)



# 训练、测试
# 对新样本，找到训练集中距离最近的 k 个样本，采用多数投票决定类别
knn = KNeighborsClassifier(n_neighbors=1)   # 获得kNN模型
knn.fit(x_train, y_train)           # 用训练集数据对模型进行训练
y_pre = knn.predict(x_test)
score = knn.score(x_test, y_test)   # 比较测试集的模型输出和真实标签，获得分数（一定程度表征正确率）
print('The score of this classifier is :', score)
getparallel(iris)            # 获取平行坐标图，查看各特征分布

'''
# 绘图观察哪个样本被错判
plt.figure()
plt.title("KNN Classification")
data_order = np.arange(1, 50+1)   # 生成1-50序列，是测试集样本序号，作为绘图的横轴
plt.scatter(data_order, y_test, color='b')  # 训练数据打点
plt.scatter(data_order, y_pre, color='r')  # 测试数据打点
plt.legend(["real", "predict"], loc='right', bbox_to_anchor=(1, 0.25))
plt.xticks(np.arange(1, 50+1, 2))
plt.yticks([0, 1, 2])
plt.grid()  # 加网格
plt.show()
'''


# 对比不同 k 下的分类效果
k_values = [1, 3, 5, 7, 9, 11, 15, 20, 25]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    scores.append(score)
    print(f"k = {k:2d}, accuracy = {score:.3f}")

# 可视化对比不同 k 值的效果
plt.figure(figsize=(8,5))
plt.plot(k_values, scores, marker='o')
plt.title("KNN Accuracy vs k")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy on Test Set")
plt.grid(True)
plt.show()


'''
k=1 时：
分类极准，但个别样本可能因为训练/测试划分导致局部边界不稳定；
k 增大时：
模型会“平均化”邻域，更稳健地捕捉总体趋势；
iris数据本身噪声极低 → 平滑后的决策边界反而更匹配真实分布；
所以测试集准确率（score）可能略微升高
但是随着k的增大会引入噪声
'''


best_k = 7
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pre = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pre)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap='Reds', fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (k={best_k})")
plt.show()

# 混淆矩阵的对角线元素表示分类正确的样本数，非对角线元素表示分类错误的样本数




'''
总预测正确数 = 16 + 18 + 15 = 49
总样本数 = 16 + 19 + 15 = 50
准确率（accuracy） ≈ 49 / 50 = 0.98 (98%)
'''