import numpy as np

a = range(-1,5)
print(list(a))

err_arr = np.mat(np.ones((5,2)))

print(err_arr)
print(type(err_arr))
print(err_arr[:,1])

c = np.mat([1.0,1.0,-1.0,-1.0,1.0]) # 转化为二维数组
print(c)
print(c.sum())  # 求元素总和

err_arr = np.mat(np.ones((5,1)))
prd = np.mat(np.zeros((5,1)))
test_labels = [0,1,0,1,1]
print(err_arr[prd != np.mat(test_labels).T])  # [[1,1,1]]????删除了??