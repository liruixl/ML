import numpy as np

pred = np.matrix([-1,1,1,-1,1,-1,1])
sorted_index  = pred.argsort()
print(sorted_index)  # [[0 3 5 1 2 4 6]]

