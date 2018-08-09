
def loadDataSet(filename):
    num_feature = len(open(filename,'r').readline().split('\t'))
    # print('特征数:',num_feature)
    data_mat = []
    label_mat = []

    with open(filename) as fr:
        for line in fr.readlines():  # 注意是readline还是reaelines
            line_arr = []
            cur_line = line.strip().split('\t')

            for i in range(num_feature - 1):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
    return data_mat,label_mat