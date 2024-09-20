from __future__ import absolute_import, division, print_function, unicode_literals
from learn_multigrid.assembly.MassMatrix import *
from learn_multigrid.solvers.Multigrid import *

import time
import scipy
import scipy.io as sio
import tensorflow as tf
import os

import numpy as np
import copy
from sklearn.model_selection import train_test_split

from tensorflow import keras
# from tensorflow.keras import layers

import matplotlib.pyplot as plt


def get_conn(matrix):
    matrix = matrix - np.diag(np.diag(matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] > 0:
                matrix[i,j] = 1
    return matrix


def coarsening(conn):
    C = []
    F = []
    R = list(range(0,len(conn)))

    while R:
        index = min(R)
        R = list(set(R) - set([index]))
        C.append(index)
        row = conn[index, :]
        row = np.where(row > 0)[0]
        neigh = list(set(row) & set(R))
        R = list(set(R)-set(neigh))
        F.extend(neigh)

    return C, F

def extract_patches(M, conn, C):
    patches = np.zeros(shape=(len(C), 7))
    neighs = np.zeros(shape=(len(C), 2))
    for i in range(len(C)):
        patch = np.zeros(7)

        # Element to coarsen
        node = C[i]

        row_M = copy.copy(M[node,:])
        d = row_M[node]
        row_M[node] = 0
        indices = np.flip(np.argsort(row_M))
        ordered = np.flip(np.sort(row_M))

        # Strongest connected neighbors
        strong = indices[:2]
        j = strong[0]
        k = strong[1]

        # Saving strongly neighbors to fill columns of B
        neighs[i, 0] = j
        neighs[i, 1] = k

        patch[3] = d

        # Entries related to strongly neighbors
        patch[2] = row_M[j]
        patch[4] = row_M[k]

        # diagonals of strongly connected
        patch[1] = M[j, j]
        patch[5] = M[k, k]

        # Entry related to strongest connection to j
        row_j = copy.copy(M[j, :])
        row_j[j] = 0
        row_j[i] = 0

        patch[0] = np.max(row_j)

        # Entry related to strongest connection to k
        row_k = copy.copy(M[k, :])
        row_k[k] = 0
        row_k[i] = 0
        patch[-1] = np.max(row_k)

        patches[i, :] = patch

    return patches, neighs.astype(int)


def prepare_input(mass):
    std = np.array([0.0004888, 0.00201229, 0.00054178, 0.00209122, 0.00052595,
                    0.00217302, 0.00058262])

    mean = np.array([6.75344883e-05, 2.73653317e-04, 6.92921701e-05, 2.76313787e-04,
                     6.88647231e-05, 2.78579309e-04, 7.04249314e-05])

    # prepare NN input
    k = 0
    data = np.zeros(shape=(int((mass.shape[0]-1) / 2) - 1, 7))

    for i in (range(1, mass.shape[0] - 2, 2)):
        locM_2 = mass[i:i + 3, i - 1:i + 4]
        locM = copy.copy(locM_2)
        locM[0, 3] = locM[0, 4] = 0
        locM[1,0] = locM[1,1] = 0
        locM[1,4] = locM[2,0] = 0
        locM[2,1] = locM[2,2] = 0
        row = locM[np.nonzero(locM)]
        # _, idx = np.unique(locM, return_index=True)
        # row = locM[np.sort(idx)]
        while len(row)<7:
            row = np.append(row,0)
        data[k, :] = row
        k = k + 1
    data = (data - mean) / std
    return data


def compute_B(dim, model, data):
    # Construct B
    B = np.zeros(shape=(dim, int((dim - 1) / 2) + 1))
    res = model.predict(data)

    i = 0
    j = 0

    for k in range(0, res.shape[0]):
        patch = res[k]
        B[i + 2, j] = patch[2]

        middle = patch[4:7]
        B[i + 1:i + 4, j + 1] = middle

        B[i + 2, j + 2] = patch[8]

        i = i + 2
        j = j + 1
    return B

def comp_B(res, M, C):
    B = np.zeros(shape=(len(M), len(C)))

    for i in range(0, len(C)):
        node = C[i]
        patch = res[i]

        center = C[i], i
        up = neighs[i, 0], i
        down = neighs[i, 1], i

        if i == 0:
            left = C[i], len(C) - 1
        else:
            left = C[i], i - 1

        if i == len(C) - 1:
            right = C[i], 0
        else:
            right = C[i], i + 1

        center_entry = patch[5]
        left_entry = patch[2]
        right_entry = patch[8]
        up_entry = patch[4]
        down_entry = patch[6]

        B[center] = center_entry
        B[left] = left_entry
        B[right] = right_entry
        B[up] = up_entry
        B[down] = down_entry
    return B


def refine(i):
    i[np.isnan(i)] = 0
    return i


path = "/Users/claudio/Desktop/PhD/Codes/MATLAB/Unregular_grids/matrices/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model('../data/models/best_penalty_2.h5', compile=False)

std = np.array([0.0004888, 0.00201229, 0.00054178, 0.00209122, 0.00052595,
                0.00217302, 0.00058262])

mean = np.array([6.75344883e-05, 2.73653317e-04, 6.92921701e-05, 2.76313787e-04,
                 6.88647231e-05, 2.78579309e-04, 7.04249314e-05])

A = sio.loadmat(path+'stiff_5.mat')
A = A['A']


M = sio.loadmat(path+'mass_5.mat')
M = M['M']

rhs = sio.loadmat(path+'rhs_5.mat')
rhs = rhs['rhs']
print("data read")
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
conn = get_conn(M)
C, F = coarsening(conn)
patches, neighs = extract_patches(M, conn, C)
patches = (patches - mean) / std
# data_M = prepare_input(M)
print("input prepared")
res = model.predict(patches)

B = comp_B(res, M, C)


# B = compute_B(M.shape[0], model, data_M)
print("B computed")
# M_sum = M.sum(axis=1)
# B_sum = B.sum(axis=1)
# MB_diff = M_sum - B_sum
# B[0:2, 0] = MB_diff[0:2]
# B[-2:, -1] = MB_diff[-2:]
B_sums = B.sum(axis=1)
i = B / B_sums[:, np.newaxis]
i = refine(i)
#
nmg = SemiGeometricMG(A,rhs, i)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=2, error=1e-10, max_iterations=40)
nmg.plot('log')
plt.show()
