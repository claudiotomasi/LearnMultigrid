from __future__ import absolute_import, division, print_function, unicode_literals
from learn_multigrid.assembly.MassMatrix import *
from learn_multigrid.solvers.Multigrid import *
from learn_multigrid.L2_projection.L2Projection import *
from learn_multigrid.assembly.LoadVector import *
from learn_multigrid.mesh.Mesh1D import *
from learn_multigrid.assembly.Quadrature import *
from learn_multigrid.assembly.ShapeFunction import *
from learn_multigrid.L2_projection.Intersection import *
from learn_multigrid.L2_projection.CouplingOperator import *
from learn_multigrid.assembly.StiffnessMatrix import *
from learn_multigrid.utilities.laplacian import *

import time
import scipy
import scipy.io as sio
import tensorflow as tf
import os

import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
# from tensorflow.keras import layers

import matplotlib.pyplot as plt


path = "../data/dataset/matrices/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('../data/models/Columnmodel_2_2.h5')

A = sio.loadmat(path+'stiff_tri.mat')
A = A['H'].toarray()


M = sio.loadmat(path+'mass_tri.mat')
M = M['M'].toarray()

rhs = sio.loadmat(path+'rhs_tri.mat')
rhs = rhs['g']

# A = A.todense()
# M = M.todense()
std = np.array([0.00089083, 0.00364883, 0.00093721, 0.00374349, 0.00093641])
mean = np.array([0.00019376, 0.00078147, 0.00019697, 0.00078759, 0.00019682])

k = 0
data_M = np.zeros(shape=(int(M.shape[0]/2) - 1, 5))



# data_M[1, :] = np.concatenate(([0], row))
for i in (range(1, M.shape[0]-2, 2)):
    locM = M[i:i+3, i-1:i+2]
    locM = locM[np.nonzero(locM)]
    _, idx = np.unique(locM, return_index=True)
    row = locM[np.sort(idx)]
    for j in range(len(row)):
        data_M[k, j] = row[j]
    k = k+1


# data_M[-2, :] = np.concatenate((row, [0]))

# Construct B
data_M = (data_M - mean)/std
# B = np.zeros(shape=(mesh.get_np(), mesh_C.get_np()))
# j = 0
# count = 0
# for i in range(2, mesh.get_np() - 2):
#     row = data_M[i, :]
#     res = model.predict(np.array([row]))
#     # print(res[0])
#     B[i, j:j+3] = res[0]
#     # print(res, i, j)
#     count += 1
#     if count == 2:
#         count = 0
#         j += 1
#
#     # i += 1
#     # row = data_M[i, :]
#     # res = model.predict(np.array([row]))
#     # B[i, j:j + 3] = res
#     # i += 1
#     # print(i)
startNN = time.time()
B = np.zeros(shape=(M.shape[0], int(M.shape[0]/2)))
res = model.predict(data_M)

i = 0
k = 0
for j in range(1, B.shape[1]-1):
    column = res[k]
    if j == B.shape[1]-2:
        column = column[1:]
        B[i:i + 5, j] = column
    else:
        B[i:i+6, j] = column
    i = i + 2
    k = k + 1

# i += 1
# row = data_M[i, :]
# res = model.predict(np.array([row]))
row = data_M[1, :]
res = model.predict(np.array([row]))
# B[1, 0:3] = res[0]
sum_row_M = M.sum(axis=1)
sum_row_B = B.sum(axis=1)
diff = sum_row_M - sum_row_B
B[0:4, 0] = diff[0:4]
B[-3:, -1] = diff[-3:]



# B[0, 0] = 1
# B[1,0] = 0.5
# B[1,1] = 0.5
# B[-2, -2] = 0.5
# B[-2, -1] = 0.5
# B[-1, -1] = 1

# Now to Q
row_sums = B.sum(axis=1)
Q = B / row_sums[:, np.newaxis]
endNN = time.time()
timeNN = endNN - startNN
# load = LoadVector(mesh)
# rhs = load.compute_rhs_1d(f)
# rhs[0] = 0
# boundaries:
# A[1,0]= 0
# A[-2, -1] = 0
# A[0,:] = 0
# A[-1,:] = 0
# A[0,0] = 1
# A[-1,-1] = 1

#ions=300)
# cg.plot('log')
#
nmg = NeuralMG(A,rhs, Q)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
nmg.plot('log')

plt.show()

print("NN: ", timeNN)