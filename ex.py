# from scipy.sparse import hstack, vstack, csr_matrix, save_npz, load_npz, coo_matrix
# A = coo_matrix([[1, 2], [3, 4]])
# B = [5, 6]
# print(A)
# print()
# A = vstack((A, B))
# print(A)
import numpy as np
import csv
import pandas

df = pandas.read_csv('Dataset/VnEmoLex.csv')
d = {}
d_rows, d_cols = df.shape
count = 0
for index in range(d_rows):
    lex = df['Vietnamese'][index].strip()
    lex = lex.replace(' ', '_')
    if df['Positive'][index] == 1:
        d[lex] = True
    else:
        d[lex] = False


print(d)
