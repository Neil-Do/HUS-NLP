# from scipy.sparse import hstack, vstack, csr_matrix, save_npz, load_npz, coo_matrix
# A = coo_matrix([[1, 2], [3, 4]])
# B = [5, 6]
# print(A)
# print()
# A = vstack((A, B))
# print(A)
import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas

m = np.array([[1,2,3],[4,5,6],[7,8,9]])
m = csr_matrix(m)
c = np.array([0,1,2])
c = csr_matrix(c)
m = m.multiply(c)


print(m.toarray())
