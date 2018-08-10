import numpy as np
import math
from loadDataSet import loadDataSet
# 基于单层决策树(树状stump)的构建的弱分类器

def loadSimpDate():
    data_mat = np.matrix([[1.,2.1],
                      [2.,1.1],
                      [1.3,1.],
                      [1.,1.],
                      [2.,1.]])
    class_labels = [1.0,1.0,-1.0,-1.0,1.0]
    return data_mat,class_labels


def stumpClassify(data_matrix,dimen,thresh_val,thresh_ineq):
    ret_array = np.ones((np.shape(data_matrix)[0],1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:,dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:,dimen] > thresh_val] = -1.0
    return ret_array  # 这里返回的不是矩阵，而是ndarray


def buildStump(data_arr,class_labels,D):
    # ``mat`` Equivalent to ``matrix(data, copy=False)``.传引用
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_matrix)

    num_step = 10.0  # 用于在特征的所有可能值遍历
    best_stump = {}
    best_class_est = np.mat(np.zeros((m,1)))
    min_error = np.inf

    for i in range(n):  # 遍历特征
        range_min = data_matrix[:,1].min()
        range_max = data_matrix[:,1].max()
        step_size = (range_max-range_min)/num_step
        # print('dim %d step = %.2f' % (i,step_size))
        for j in range(-1,int(num_step)+1):  # 分步遍历特征值
            for inequal in ['lt','gt']:
                thresh_val = (range_min + float(j)*step_size)
                predicted_vals = stumpClassify(data_matrix,
                                               i,thresh_val,
                                               inequal)
                err_arr = np.mat(np.ones((m,1)))
                err_arr[predicted_vals == label_mat] = 0
                weight_error = D.T*err_arr  # 这里就是矩阵乘了

                # print('split:dim %d, thresh %.2f, thresh inequal: %s,'
                #       'the weighted error is %.3f' % \
                #       (i, thresh_val,inequal,weight_error))
                if weight_error<min_error:
                    min_error = weight_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump,min_error,best_class_est


def asaBoostingTrainDS(data_arr,class_labels,num_it=40):
    weak_class_est = []  # 存储弱分类器
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m,1))/m)
    agg_class_est = np.mat(np.zeros((m,1)))
    for i in range(num_it):

        # print('D:', D.T)
        best_stump, error, class_est = buildStump(data_arr,class_labels,D)

        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        best_stump['alpha'] = alpha

        weak_class_est.append(best_stump)
        # print('classEst:', class_est)

        expon = np.multiply(-1*alpha*np.mat(class_labels).T,class_est)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()  # 不是元素个数，而是矩阵元素求和，使D成为一个分布

        agg_class_est += class_est
        # print('aggClassEst:',agg_class_est.T)

        # ``sign`` returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m,1)))
        error_rate = agg_errors.sum()/m
        print('total error;', error_rate,'\n')

        if error_rate == 0:
            print('error rate = 0,结束训练')
            break
    return weak_class_est, agg_class_est  # 返回弱分类器列表和类预测估计


def adaClassify(dattocalss,classifier_arr):
    data_matrix = np.mat(dattocalss)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m,1)))

    for i in range(len(classifier_arr)):
        class_est = stumpClassify(data_matrix,classifier_arr[i]['dim'],
                                  classifier_arr[i]['thresh'],
                                  classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha']*class_est
        # print('预测是：',agg_class_est)
    return np.sign(agg_class_est)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    y_sum = 0.0  # 用于计算AUC的值，曲线下面积
    num_pos_class = sum(np.array(classLabels) == 1.0)  # 数组过滤，计算正例数目
    y_step = 1/float(num_pos_class)
    x_step = 1/float(len(classLabels)-num_pos_class)

    sorted_indicies = predStrengths.argsort()

    fig = plt.figure()  # 构建画笔
    fig.clf()
    ax = plt.subplot(111)

    for index in sorted_indicies.tolist()[0]:  # 这里是二维矩阵，形状是1*n，所以要0索引
        if classLabels[index] == 1.0:
            delx = 0
            dely = y_step
        else:
            delx = x_step
            dely = 0
            y_sum += cur[1]
        ax.plot([cur[0],cur[0]-delx],[cur[1],cur[1]-dely],c = 'b')
        cur = (cur[0]-delx,cur[1]-dely)
    ax.plot([0,1],[0,1],'b--') # 虚线
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('ROC')
    ax.axis([0,1,0,1])
    plt.show()
    print('AUC is ',y_sum*x_step)



if __name__ == '__main__':
    # data_mat, class_labels = loadSimpDate()
    data_list, class_labels = loadDataSet('horseColicTraining2.txt')
    data_mat = np.matrix(data_list)
    # D = np.mat(np.ones((5,1))/5)
    # print(D)
    # best_stump, min_error, best_class_est = buildStump(data_mat,class_labels,D)
    # print(best_stump)
    # print(best_class_est)
    classifer, agg_class_est = asaBoostingTrainDS(data_mat,class_labels,10)
    # print(adaClassify([[0,0],[5,5]],classifer))

    # test_arr, test_labels = loadDataSet('horseColicTest2.txt')
    # prd = adaClassify(test_arr,classifer)
    # # print(len(test_labels)) # 67
    # err_arr = np.mat(np.ones((len(test_labels),1)))
    #
    # print(prd)
    #
    # print(err_arr[prd != np.mat(test_labels).T].sum())  # 错误数量

    plotROC(agg_class_est.T,class_labels)






