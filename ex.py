# from scipy.sparse import hstack, vstack, csr_matrix, save_npz, load_npz, coo_matrix
# A = coo_matrix([[1, 2], [3, 4]])
# B = [5, 6]
# print(A)
# print()
# A = vstack((A, B))
# print(A)
import numpy as np


a = np.ones(3)
b = np.zeros(4)
c = np.ones(3) * (-1)
d = np.hstack((a, b, c))
# d.resize(len(d))
print(d)
