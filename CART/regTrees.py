import numpy as np
from datafactory import loadDataSet

'''用字典保存树的数据结构：
待切分的特征
带切分的特征值
右子树。当不需要切分也可以是单个值
左子树
'''


def binSplitDataSet(dataSet, feature, value):
    """
    函数说明:根据特征阈值切分数据集合
    Parameters:
        dataSet - 数据集合np.array  np.matrix
        feature - 带切分的特征，以index表示
        value - 特征阈值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    # 生成叶结点，返回目标变量标签的均值
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    # 方差损失，为什么要乘以数据数量呢? 原公式为平方差的和，为方差除以了N
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树,字典
    """
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果没有特征,则返回特征值，即递归到叶节点了
    if feat is None:
        return val
    # 回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    函数说明:找到数据的最佳二元切分方式函数
    Parameters:
        dataSet - 数据集合
        leafType - 生成叶结点
        regErr - 误差估计函数
        ops - 用户定义的参数构成的元组
    Returns:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
    """
    # tolS允许的误差下降值,tolN切分的最少样本数
    tolS = ops[0]
    tolN = ops[1]  # 节点最少数据量
    # 如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 统计数据集合的行m和列n
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    # 遍历所有特征列
    for featIndex in range(n - 1):  # 标签除外：-1
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 1.如果数据少于tolN,则退出本次阈值
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 计算误差估计
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 2.如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 3.如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue

# ========================减枝处理==============================
def isTree(obj):
    return type(obj).__name__ =='dict'
def getMean(tree):
    # 函数说明:对树进行递归塌陷处理(即返回树平均值，合并左右子树为一个值)
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0
def prune(tree, testData):
    '''
    返回的是一棵树或是一个均值
    '''
    # 如果测试集为空,则对树进行塌陷处理
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 如果有左子树或者右子树,则切分数据集
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 处理左子树(剪枝)
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 处理右子树(剪枝)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 如果当前结点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(
            np.power(rSet[:, -1] - tree['right'], 2))
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            # print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

if __name__ == '__main__':
    data_path = 'prune_train.txt'
    myDat = loadDataSet(data_path) # y是标签，连续性
    myMat = np.mat(myDat)
    print('before pruning:')
    tree = createTree(myMat)
    print(tree)

    test_filename = 'prune_test.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print('剪枝后:')
    new_tree = prune(tree, test_Mat)
    print(new_tree)