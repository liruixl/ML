import math
import operator


def createDataSet():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    label = ['in water', 'flippers']
    return dataset, label


def Ent(dataset):
    """
    计算数据集香农熵的值
    :param dataset: 包含特征和标签的数据集
    :return: 熵
    """
    numEntries = len(dataset)
    labelCounts = {}
    for feature in dataset:
        currentLabel = feature[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0
    for key in labelCounts:
        num = labelCounts[key]
        prop = float(num)/numEntries
        entropy = entropy - prop*math.log(prop, 2)
    return entropy


def splitDataSet(dataset,axis,value):
    retDataset = []
    for feature in dataset:
        if feature[axis] == value:
            reduced_feat = feature[:axis]
            reduced_feat.extend(feature[axis+1:])
            retDataset.append(reduced_feat)
    return retDataset


def chooseBestFeatureToSplit(dataset):
    '''ID3算法
    
    '''
    numFeatures = len(dataset[0]) - 1
    base_entropy = Ent(dataset)
    best_info_gain = 0.0
    beat_feature = -1

    for i in range(numFeatures):
        feat_val_list = [feature[i] for feature in dataset]
        feat_val_set = set(feat_val_list)

        new_entropy = 0.0
        for value in feat_val_set:
            sub_dataset = splitDataSet(dataset,i,value)
            proportion = float(len(sub_dataset))/len(dataset)
            new_entropy += proportion*Ent(sub_dataset)
        info_gain = base_entropy - new_entropy

        if(info_gain>best_info_gain):
            best_info_gain = info_gain
            beat_feature = i
    return beat_feature


def majorityClass(class_list):
    classCount = {}
    for vote in class_list:
        if vote not in class_list.key():
            class_list[vote] = 0
        class_list[vote] += 1

    # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
    # list 的 sort 方法返回的是对已经存在的列表进行操作，
    # 而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset,labels):
    class_list = [feature[-1] for feature in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majorityClass(class_list)

    best_feature = chooseBestFeatureToSplit(dataset)
    best_feature_label = labels[best_feature]

    my_tree = {best_feature_label:{}}
    del labels[best_feature]
    feat_val = [feature[best_feature] for feature in dataset]
    feat_val_set = set(feat_val)
    for value in feat_val_set:
        sub_labels = labels[:]  # 为什么不是sub_labels = labels
        my_tree[best_feature_label][value] = createTree(splitDataSet(dataset,best_feature,value),
                                                        sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    #  使用已有的决策树分类未知类别的数据
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)

    for key in second_dict.key():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classify(second_dict[key],feat_labels,test_vec)  # 递归检查特征值
            else:
                class_label = second_dict[key]
    return class_label


def storeTree(input_tree,filename):
    import pickle
    with open(filename,'wb') as fw:
        pickle.dump(input_tree,fw)

def getTree(filename):
    import pickle
    with open(filename,'rb') as fr:
        return pickle.load(fr,)


if __name__ == '__main__':
    dataset, labels = createDataSet()
    # ent = Ent(dataset)
    # print(ent)

    # feat = chooseBestFeatureToSplit(dataset)
    # print(feat)
    # print(labels[feat])

    my_tree = createTree(dataset,labels)
    print(my_tree)
    storeTree(my_tree,'tree.txt')
    print(getTree('tree.txt'))