import numpy as np

dataSet = np.array([[1, 0, 0],
                    [0, 2, 0],
                    [1, 1, 0]])
'''mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
 1. a = dataSet[:,1]>0  # <class 'numpy.ndarray'> [0 2 1]=== >1===→[False  True True]  (3,)这不是(3,1)
    a = dataSet[:, 0:2]  # (3,2)
 2.np.nonzero([False,True, True])  # (array([1,2], dtype=int64),)
例如：
   np.nonzero(dataSet)
   (array([0, 1, 2, 2]), array([0, 1, 0, 1])) # tuple, 每个维度下的索引,二维可以理解为每个array存储x，y坐标
   3. np.nonzero([False  True True])[0]  # 由于第二步输出是tuple，[0]表示取第一个维度的索引。
   4.dataSet[np.nonzero([False ,True, True])[0], :]
     dataSet[[1,2], :]
     dataSet[1:, :]  # 一样的效果，切片
'''

# a = dataSet[:, 0:2]
# print(a)
# print(type(a))
print(dataSet[np.nonzero([False, True, True])[0], :])
print(dataSet[[1, 2], :])
print(dataSet[1:, :])



