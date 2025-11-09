import math

# 创建数据
def createDataSet():
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]

    labels = ['age', 'job', 'house', 'loan']  # 特征标签
    return dataSet, labels

# 获取当前样本最多的标签
def getMaxLabelByDataSet(curLabelList):
    classCount = {}
    maxKey, maxValue = None, None
    for label in curLabelList:
        if label in classCount.keys():
            classCount[label] += 1
            if maxValue < classCount[label]:
                maxKey, maxValue = label, classCount[label]
        else:
            classCount[label] = 1
            if maxKey is None:
                maxKey, maxValue = label, 1
    return maxKey

# 计算信息熵
def calcEntropy(dataSet):
    # 获取样本数
    exampleNum = len(dataSet)

    # 计算每个标签出现的数量
    labelCount = {}
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel in labelCount.keys():
            labelCount[curLabel] += 1
        else:
            labelCount[curLabel] = 1
    # 计算信息熵
    entropy = 0.0
    for key, value in labelCount.items():
        # 计算标签的概率
        prob = labelCount[key] / exampleNum
        # 计算信息熵
        curEntropy = -prob * math.log(prob, 2)
        entropy += curEntropy
    return entropy


# 按照某个特征划分数据集
def chooseBestFeatureToSplit(dataSet):
    # 计算特征数量（减去最后一列的标签）
    featureNum = len(dataSet[0]) - 1

    # 计算划分前的信息熵
    curEntropy = calcEntropy(dataSet)

    # 找最好的特征划分
    bestInfoGain = 0.0 # 最大化的信息增益
    bestFeatureIndex = -1 # 最优特征的索引

    for i in range(featureNum):
        # 获取当前列的特征
        featList = [example[i] for example in dataSet]

        # 获取唯一值
        uniqueVals = set(featList)

        # 计算划分后的信息熵
        newEntropy = 0.0

        #计算不同特征划分的熵值
        for val in uniqueVals:
            # 根据当前特征划分dataset
            subDataSet = splitDataSet(dataSet, i, val)
            # 计算加权概率值
            weight = len(subDataSet) / float(len(dataSet))
            newEntropy += (calcEntropy(subDataSet) * weight)
        # 计算信息增益
        infoGain = curEntropy - newEntropy
        # 更新最大的信息增益
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i

    return bestFeatureIndex


def splitDataSet(dataSet, featureIndex, value):
    returnDataSet = []
    for featVec in dataSet:
        if featVec[featureIndex] == value:
            # 去掉featureIndex特征
            deletedFeatVec = featVec[:featureIndex]
            deletedFeatVec.extend(featVec[featureIndex + 1:])
            returnDataSet.append(deletedFeatVec)
    return returnDataSet

# 创建决策树
def createTreeNode(dataSet, labels, featLabels): # labels是特征标签，featLabels是用来存储选择的最优特征标签

    # 取出当前节点的样本标签
    curLabelList = [example[-1] for example in dataSet]

    # ------------停止条件--------------
    # 1. 判断当前节点的样本的标签是不是已经全为1个值了，如果是则直接返回其唯一类别
    if len(curLabelList) == curLabelList.count(curLabelList[0]):
        return curLabelList[0]  # 假设curLabelList = ['yes', 'yes', 'yes'],满足调价则标签纯净
    # 2. 判断当前可划分的特征数是否为1，如果为1则直接返回当前样本里最多的标签
    if len(labels) == 1:
        return getMaxLabelByDataSet(curLabelList)
    
    # ------------正常选择特征划分的步骤-------------
    # 1. 选择最优特征
    bestFeatIndex = chooseBestFeatureToSplit(dataSet)
    # 2. 获取最优特征的标签
    bestFeatLabel = labels[bestFeatIndex] # 返回的是特征的标签，即'age','job','house','loan'之一
    # 3. 将特征划分加入当前决策树
    featLabels.append(bestFeatLabel)
    # 4. 构造当前节点
    myTree = {bestFeatLabel: {}}
    # 5. 去掉已经划分的特征
    del labels[bestFeatIndex]
    # 6. 获取当前最优特征的所有取值，用当前特征进行划分
    featValues = [example[bestFeatIndex] for example in dataSet]
    # 7. 获取唯一值
    uniqueVals = set(featValues)
    # 8. 递归构造决策树
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTreeNode(
            splitDataSet(dataSet, bestFeatIndex, value), labels.copy(), featLabels.copy())
    return myTree




# 测试一下！！！
# 1. 获取数据集
dataSet,labels = createDataSet()
# 2. 构建决策树
myDecisionTree = createTreeNode(dataSet,labels,[])
# 3. 输出
print(myDecisionTree)
