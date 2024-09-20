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


def extract_patches(M, C):
    patches = np.zeros(shape=(len(C), 43))
    neighbours = np.ones(shape=(len(C), 6), dtype=int)*(-1)
    for i in range(0,len(C)):
        node = C[i]
        patch = np.zeros(43)

        # Diagonal entry
        diag_M = M[node,node]
        patch[0] = diag_M
        row = copy.copy(M[node, :])
        row[node] = 0
        neighs = np.where(row)[0]

        # Row entries
        patch[1:1+len(neighs)] = row[np.nonzero(row)]
        indexes_neigh =  [7, 13, 19, 25, 31, 37]
        for j in range(0,len(neighs)):
            node_neigh = neighs[j]
            neigh_entries = np.zeros(6)
            neigh_entries[0] = M[node_neigh, node_neigh]
            row_neigh = copy.copy(M[node_neigh, :])
            row_neigh[node_neigh] = 0
            row_neigh[node] = 0
            data = row_neigh[np.nonzero(row_neigh)]
            neigh_entries[1:1 + len(data)] = data
            patch[indexes_neigh[j]:indexes_neigh[j]+6] = neigh_entries

        patches[i, :] = patch
        # coarse_neigh = list(set(neighs).intersection(F))
        neighbours[i,0:len(neighs)] = neighs
    return patches, neighbours


path = "/Users/claudio/Desktop/PhD/Codes/MATLAB/Unregular_grids/matrices/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model('../data/models/2D_column_WITHOUT_distance.h5', compile=False)

std = np.array([0.0056043 , 0.00105846, 0.00101318, 0.00101015, 0.00105614,
               0.00100385, 0.00100645, 0.00580584, 0.00105955, 0.00102391,
               0.00102468, 0.0010217 , 0.00101963, 0.00553394, 0.00100468,
               0.00094116, 0.00101921, 0.00102359, 0.00093858, 0.00554135,
               0.00101274, 0.00094689, 0.0009389 , 0.00102127, 0.00102082,
               0.00579647, 0.00105001, 0.00101871, 0.00102138, 0.00102781,
               0.00101766, 0.00553096, 0.00100853, 0.00102061, 0.001021,
               0.00094363, 0.00093958, 0.00553589, 0.00100309, 0.00102561,
               0.00093998, 0.00094611, 0.00101479])

mean = np.array([0.00181986, 0.00030311, 0.00030358, 0.00030361, 0.00030359,
               0.00030287, 0.0003031 , 0.00181859, 0.00030356, 0.00030324,
               0.00030294, 0.00030311, 0.00030263, 0.00181836, 0.00030261,
               0.00030281, 0.0003027 , 0.0003034 , 0.00030326, 0.00182142,
               0.00030396, 0.00030431, 0.00030277, 0.00030345, 0.00030332,
               0.00182119, 0.0003035 , 0.00030274, 0.00030318, 0.00030455,
               0.00030362, 0.00182115, 0.00030389, 0.00030302, 0.00030311,
               0.00030441, 0.00030385, 0.00182235, 0.0003034 , 0.00030415,
               0.00030359, 0.00030436, 0.00030375])

A = sio.loadmat(path+'stiff_4.mat')
A = A['A']


M = sio.loadmat(path+'mass_4.mat')
M = M['M']

rhs = sio.loadmat(path+'rhs_4.mat')
rhs = rhs['rhs']
print("data read")

conn = get_conn(M)
C, F, C_neigh, F_neigh = coarsening(conn)

# C = np.arange(0,len(M),2)
# F = np.arange(1,len(M),2)

patches, neighbours = extract_patches(M, C)
patches = (patches - mean) / std

res = model.predict(patches)

B = np.zeros((len(M),len(C)))
for i in range(0,len(C)):
    pos = neighbours[i,:]
    many = len(np.where(pos>0)[0])
    pos = pos[0:many]
    col = res[i][0:many]
    B[pos, i] = col
#
# row_sums = B.sum(axis=1)
# Q = B / row_sums[:, np.newaxis]
#
# nmg = SemiGeometricMG(A,rhs, Q)
# nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=2, error=1e-10, max_iterations=40)
# nmg.plot('log')
# plt.show()