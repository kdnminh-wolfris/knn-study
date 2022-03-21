import pandas as pd
import numpy as np

def read_data(faiss_out, my_out, inp):
    faiss_knn = pd.read_csv(faiss_out, header=None, sep=' ').to_numpy()
    my_knn = pd.read_csv(my_out, header=None, sep=' ').to_numpy()
    inp_matrix = pd.read_csv(inp, skiprows=1, header=None, sep=' ').to_numpy()

    faiss_knn = faiss_knn[:, 1:-1].astype('int32')
    my_knn = my_knn[:, :-1].astype('int32')
    inp_matrix = inp_matrix[:, :-1]

    return faiss_knn, my_knn, inp_matrix

def distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) * (a[i] - b[i])
    return dist**0.5

def distance_check():
    error = 0
    for i in range(my_knn.shape[0]):
        for j in range(my_knn.shape[1]):
            error += np.abs(
                distance(inp_matrix[i], inp_matrix[faiss_knn[i, j]])
                - distance(inp_matrix[i], inp_matrix[my_knn[i, j]])
            )
    print('Neighbour distances error:', error)

def similarity_check():
    cnt = 0
    for i in range(my_knn.shape[0]):
        faiss_knn[i] = np.sort(faiss_knn[i])
        my_knn[i] = np.sort(my_knn[i])
        j = 0
        for v in my_knn[i]:
            while j < faiss_knn.shape[1] and faiss_knn[i][j] < v:
                j += 1
            if j == faiss_knn.shape[1] or v != faiss_knn[i, j]:
                cnt += 1
    print('Number of different neighbours of every point:', cnt)

#-----MAIN-----
faiss_knn, my_knn, inp_matrix = read_data('knn_index.out', 'GSE128223.out', 'GSE128223.inp')
print('Faiss knn shape:', faiss_knn.shape)
print('My knn shape:', my_knn.shape)
print('Input matrix shape:', inp_matrix.shape)
similarity_check()