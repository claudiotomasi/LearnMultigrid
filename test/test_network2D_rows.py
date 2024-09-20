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
    CC = []
    FF = []
    R = list(range(0,len(conn)))
    C_neighs = {}
    F_neighs = {}
    while R:
        index = min(R)
        R = list(set(R) - {index})
        CC.append(index)
        row = conn[index, :]
        row = np.where(row > 0)[0]
        C_neighs[index] = row
        neigh = list(set(row) & set(R))
        R = list(set(R)-set(neigh))

        rows = conn[neigh, :]
        for j in range(0,len(rows)):
            neigh_row = rows[j]
            F_neighs[neigh[j]] = np.where(neigh_row > 0)[0]
        FF.extend(neigh)

    return CC, FF, C_neighs, F_neighs


def extract_patches(M, C, F):
    FF = list(F)
    # list(set(F) - {1})
    patches = np.zeros(shape=(len(C), 13))
    neighbors = np.ones(shape=(len(F), 4))*(-1)
    for i in range(0,len(F)):
        node = F[i]
        patch = np.zeros(13)

        # Diagonal entry
        patch[0] = M[node,node]

        row = copy.copy(M[node,:])
        row[node] = 0
        neighs = np.where(row)[0]

        # Row entries
        patch[1:1+len(neighs)] = row[np.nonzero(row)]

        for j in range(0,len(neighs)):
            patch[7+j] = M[j,j]
        coarse_neigh = list(set(neighs).intersection(C))
        neighbors[i,0:len(coarse_neigh)] = coarse_neigh
    return patches, neighbors


path = "/Users/claudio/Desktop/PhD/Codes/MATLAB/Unregular_grids/matrices/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model('../data/models/row_2d.h5', compile=False)

std = np.array([0.00560492, 0.00103963, 0.00100227, 0.00099651, 0.00098832,
       0.00099908, 0.00097832, 0.00390447, 0.00452033, 0.00516859,
       0.00437744, 0.00484039, 0.00506306])

mean = np.array([0.00182256, 0.00030696, 0.00030066, 0.00030411, 0.00030196,
       0.00030528, 0.00030359, 0.00132756, 0.00149446, 0.00166886,
       0.00150224, 0.00157559, 0.00163998])

A = sio.loadmat(path+'stiff_100.mat')
A = A['A']


M = sio.loadmat(path+'mass_100.mat')
M = M['M']

rhs = sio.loadmat(path+'rhs_100.mat')
rhs = rhs['rhs']
print("data read")

conn = get_conn(M)
C, F, C_neigh, F_neigh = coarsening(conn)

# C = np.arange(0,len(M),2)
# F = np.arange(1,len(M),2)

patches, neighbors = extract_patches(M, C, F)
patches = (patches - mean) / std

res = model.predict(patches)
#
# B = np.zeros((len(M),len(C)))
# for i in range(0,len(C)):
#     pos = neighs[i,:]
#     entries = res[i][4:9]
#     B[pos,i] = entries
#
# row_sums = B.sum(axis=1)
# Q = B / row_sums[:, np.newaxis]
#
# nmg = SemiGeometricMG(A,rhs, Q)
# nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=2, error=1e-10, max_iterations=40)
# nmg.plot('log')
# plt.show()