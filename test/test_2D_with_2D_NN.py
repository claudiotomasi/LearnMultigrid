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


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def get_conn(mat):
    matrix = copy.copy(mat)
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
    CC_neighs = {}
    FF_neighs = {}
    while R:
        index = min(R)
        R = list(set(R) - {index})
        CC.append(index)
        row = conn[index, :]
        row = np.where(row > 0)[0]
        CC_neighs[index] = row
        neigh = list(set(row) & set(R))
        R = list(set(R)-set(neigh))

        rows = conn[neigh, :]
        for j in range(0,len(rows)):
            neigh_row = rows[j]
            FF_neighs[neigh[j]] = np.where(neigh_row > 0)[0]
        FF.extend(neigh)

    return CC, FF, CC_neighs, FF_neighs


def extract_patches(CC, mat):
    # For each c_node I need to save the index where to fill B.
    # I need 'where' -> direct neighbours of c_node in fine grid
    # And 'where_nOfN -> direct neighbours of c_node in coarse grid
    #   it contains the neighbors of the neighbors of node
    #   If we intersect with c_index, we get the correct column indexes to fill row_B

    # Then I need 'intersecting' -> rows intersecting the column of B
    _patches = np.zeros((len(CC), 43))
    _fill_idxs = np.zeros((len(CC), 31), dtype = int)
    ii = 0
    for c_node in CC:
        # patch consists of:
        # 1st is node_M - in pos 0
        # 2nd is row - in pos [1:7] (meaning until patch[6])
        # 3rd is neighs - pos [7:]
        # neighs consists of six blocks where:
        # 1st is neigh_node
        # 2nd row_neigh
        patch = np.ones(43) * -1
        fill_idx = np.ones(31) * -1
        # we need to use copy.copy(), otherwise we modify actual M
        row = copy.copy(mat[c_node, :])
        node_M = row[c_node]
        row[c_node] = 0
        where = np.where(row)[0]
        # print(where)
        row = row[row > 0]
        neighs = np.ones(36) *-1
        jj = 0
        B = []
        for neighbour in where:
            row_neigh = copy.copy(mat[neighbour, :])
            row_neigh[c_node] = 0
            B.extend(np.where(row_neigh)[0])
            neigh_node = row_neigh[neighbour]
            row_neigh[neighbour] = 0
            row_neigh = row_neigh[row_neigh > 0]
            neighs[jj] = neigh_node
            neighs[jj + 1: jj + len(row_neigh) + 1] = row_neigh
            jj = jj + 6

        # This remove duplicates maintaining original order
        B = sorted(set(B), key=B.index)
        # print(B)
        D = diff(diff(B, where), [c_node])
        # _auxset = set(D)
        where_nOfN = [x for x in D if x in CC]

        # print(where_nOfN)

        patch[0] = node_M
        patch[1: len(row) + 1] = row
        patch[7:] = neighs

        _patches[ii, :] = patch

        fill_idx[0] = c_node
        fill_idx[1:len(where) + 1] = where
        fill_idx[7: 7 + len(where_nOfN)] = where_nOfN

        # For each neighbour of c_node, i.e, for each index in 'where', we need 3 intersecting values
        # For c_node, its row was given by taking neighs(neighs(c_node)), minus 'where' minus 'c_node'
        # We need to do the same but for neighbour(c_node), thus for each element in 'where'
        # So, we start from nOfN which represents the neigh(where) and keep asking for its neighbours
        pos = 13
        for el in where:
            row_el = copy.copy(mat[el, :])
            row_el[c_node] = 0
            row_el[el] = 0
            row_neigh = np.where(row_el)[0]
            where_with_coarse = []
            for kk in row_neigh:
                neigh_row = copy.copy(mat[kk, :])
                where_with_coarse.extend(np.where(neigh_row)[0])
            where_with_coarse = sorted(set(where_with_coarse), key=where_with_coarse.index)
            where_with_coarse = [x for x in where_with_coarse if x in CC]
            where_with_coarse = diff(where_with_coarse, [c_node])
            fill_idx[pos: pos+len(where_with_coarse)] = where_with_coarse
            pos = pos + 3

        _fill_idxs[ii, :] = fill_idx

        ii = ii + 1
    return _patches, _fill_idxs


def fill_B(_res, _idx_fill, total_size):
    _B = np.zeros((total_size, len(res)))
    for j in range(0, len(res)):
        pred = _res[j]
        where = _idx_fill[j]

        node = where[0]
        col_B = where[1:7]
        col_B = col_B[col_B >= 0]
        row_B = where[7:13]
        row_B = row_B[row_B >= 0]
        # inters = where[13:]

        _B[node,node] = pred[0]
        _B[col_B,node] = pred[1:1 + len(col_B)]
        _B[node,row_B] = pred[7:7 + len(row_B)]

        pos = 13
        for k in range(0, len(col_B)):
            true_where = where[pos: pos+3][where[pos: pos+3] >= 0]
            _B[col_B[k], true_where] = pred[pos:pos+len(true_where)]
            pos = pos + 3

    return _B


path = '/Users/claudio/Desktop/PhD/Codes/MATLAB/2D/new_mesh_definition/'
problem = sio.loadmat(path+'problem.mat')
A = problem['A']
M = problem['M']
rhs = problem['rhs']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path_model = path = '/Users/claudio/Desktop/Testing_NNs/2D/Linea_and_cuthill/ne_16_5K_plus16_3Kforeach/'
model = tf.keras.models.load_model(path + '2D_6.h5', compile=False)
std = np.array([5.53811987e-04, 8.29477653e-05, 8.15864932e-05, 1.02757460e-04,
       8.32145267e-05, 1.19921640e-04, 1.03157251e-04, 5.30561121e-04,
       9.81150579e-05, 8.75618156e-05, 9.89874695e-05, 8.24586254e-05,
       8.90649957e-05, 5.53690428e-04, 1.02310745e-04, 1.20412573e-04,
       8.24586254e-05, 1.03162763e-04, 8.25537043e-05, 5.41412630e-04,
       8.77121322e-05, 8.90649957e-05, 8.38125780e-05, 8.44268506e-05,
       1.02350333e-04, 5.36701532e-04, 8.25537043e-05, 8.65000458e-05,
       1.02513174e-04, 8.77346191e-05, 1.02798640e-04, 5.58678409e-04,
       1.02350333e-04, 8.56693780e-05, 1.02435345e-04, 8.42414055e-05,
       8.25250378e-05, 5.36711365e-04, 8.77346191e-05, 1.02435345e-04,
       8.20443661e-05, 8.73433035e-05, 8.23681409e-05])
mean = np.array([1.60394913e-04, 2.62367955e-05, 2.71816357e-05, 2.64930798e-05,
       2.62623963e-05, 2.76982647e-05, 2.65227411e-05, 1.55459126e-04,
       2.59826757e-05, 2.53353773e-05, 2.61573899e-05, 2.62501679e-05,
       2.54967191e-05, 1.60895248e-04, 2.65058913e-05, 2.79384745e-05,
       2.62501679e-05, 2.67600969e-05, 2.62589815e-05, 1.56624293e-04,
       2.68071655e-05, 2.54967191e-05, 2.63061552e-05, 2.50119012e-05,
       2.65092722e-05, 1.56060721e-04, 2.62589815e-05, 2.50770406e-05,
       2.63751781e-05, 2.53962008e-05, 2.66909234e-05, 1.61060741e-04,
       2.65092722e-05, 2.65759032e-05, 2.65168970e-05, 2.75042012e-05,
       2.62562026e-05, 1.55951518e-04, 2.53962008e-05, 2.65168970e-05,
       2.61348568e-05, 2.53181609e-05, 2.60626610e-05])

connections = get_conn(M)
C, F, C_neighs, F_neighs = coarsening(connections)

patches, idx_fill = extract_patches(C, M)
#
# print(patches)

for p in patches:
    for i in range(0,len(p)):
        if p[i] < 0: p[i] = patches[7][i]  # cause 7th is full one
patches = (patches - mean) / std
res = model.predict(patches)

B = fill_B(res, idx_fill, len(M))
row_sums = B.sum(axis=1)
Q = B / row_sums[:, np.newaxis]

nmg = SemiGeometricMG(A, rhs, Q)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
nmg.plot('log')
plt.show()

# TODO: Problem - with zeros, a lot of elements are predicted with zerso, even the B(node,node)