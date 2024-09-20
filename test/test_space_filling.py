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
# from sklearn.model_selection import train_test_split

from tensorflow import keras
# from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow.keras.losses
import tensorflow.keras.backend as K

# @tf.function
# def custom_loss(y_true, y_pred):
#     scale = tf.constant([10., 1., 1., 1., 10.])
#     true = tf.math.multiply(y_true, scale)
#     pred = tf.math.multiply(y_pred, scale)
#
#     return K.mean(K.square(true - pred))



def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model = tf.keras.models.load_model('../data/models/new_coarse.h5',  custom_objects={'custom_loss': custom_loss})
model = tf.keras.models.load_model('../data/models/best_penalty_11.h5', compile=False)

path = '/Users/claudio/Desktop/PhD/Codes/MATLAB/2D/tests_on_filling_curve/'

A = sio.loadmat(path+'stiff.mat')
A = A['A']


M_h = sio.loadmat(path+'mass_hilbert.mat')
M_h = M_h['M_h']

M_m = sio.loadmat(path+'mass_morton.mat')
M_m = M_m['M_m']

reset_h = sio.loadmat(path+'reset_hilbert.mat')
reset_h = reset_h['reset_h'][0] - 1
odd_h = reset_h[::2]
reset_h_b = np.argsort(odd_h)

reset_m = sio.loadmat(path+'reset_morton.mat')
reset_m = reset_m['reset_m'][0] - 1
odd_m = reset_m[::2]
reset_m_b = np.argsort(odd_m)

rhs = sio.loadmat(path+'rhs.mat')
rhs = rhs['rhs']

# A = M_h[np.ix_(reset_h,reset_h)]


std = np.array([0.00051472, 0.00209519, 0.00055687, 0.00216233, 0.0005473 ,
       0.00224975, 0.00060137])
mean = np.array([3.92511629e-05, 1.59982482e-04, 4.07400779e-05, 1.62382236e-04,
       4.04510401e-05, 1.65230747e-04, 4.21643334e-05])

# prepare NN input
k = 0
data_M = np.zeros(shape=(int((len(M_h)-1)/2) - 1, 7))
#
for i in (range(1, len(M_h)-2, 2)):
    locM = M_h[i:i+3, i-1:i+4]
    row = np.array([locM[0,0], locM[0,1], locM[0,2], locM[1,1], locM[1,2], locM[2,2], locM[2,3]])
    # locM = locM[np.nonzero(locM)]
    # _, idx = np.unique(locM, return_index=True)
    # row = locM[np.sort(idx)]
    data_M[k, :] = row
    k = k+1
#
# Construct B
data_M = (data_M - mean)/std
#
startNN = time.time()
B_h = np.zeros(shape=(M_m.shape[0], int((M_m.shape[0]-1)/2)+1))
res = model.predict(data_M)

i = 0
j = 0

# Left Boundary
# patch = res[0, :]
# B[i:i+3, j] = patch[:3]
# B[i:i+5, j+1] = patch[3:8]
# B[i+2, j+2] = patch[8]
# #
# # i = i + 2
# # j = j + 1
#
for k in range(0, res.shape[0]):
    patch = res[k]
    # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
    B_h[i + 2, j] = patch[2]

    middle = patch[4:7]
    # middle[0] = np.mean([B[i, j + 1], middle[0]])
    B_h[i+1:i + 4, j + 1] = middle

    B_h[i + 2, j + 2] = patch[8]

    i = i + 2
    j = j + 1
#
# Right Boundary
# patch = res[-1, :]
# # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
# B[i + 2, j] = patch[2]
#
# middle = patch[3:8]
# # middle[0] = np.mean([B[i, j + 1], middle[0]])
# B[i:i+5, j+1] = patch[3:8]
#
# B[i+2:i+5, j+2] = patch[8:]


sum_row_M = M_h.sum(axis=1)
sum_row_B = B_h.sum(axis=1)
diff = sum_row_M - sum_row_B
B_h[0:2, 0] = diff[0:2]
B_h[-2:, -1] = diff[-2:]
#
#
#
# # B[0, 0] = 1
# # B[1,0] = 0.5
# # B[1,1] = 0.5
# # B[-2, -2] = 0.5
# # B[-2, -1] = 0.5
# # B[-1, -1] = 1
#
# Now to Q

B_h = B_h[np.ix_(reset_h, reset_h_b)]

# row_sums = B_h.sum(axis=1)
# Q_h = B_h / row_sums[:, np.newaxis]




# prepare NN input
k = 0
data_M = np.zeros(shape=(int((len(M_m)-1)/2) - 1, 7))
#
for i in (range(1, len(M_h)-2, 2)):
    locM = M_m[i:i+3, i-1:i+4]
    row = np.array([locM[0,0], locM[0,1], locM[0,2], locM[1,1], locM[1,2], locM[2,2], locM[2,3]])
    # locM = locM[np.nonzero(locM)]
    # _, idx = np.unique(locM, return_index=True)
    # row = locM[np.sort(idx)]
    data_M[k, :] = row
    k = k+1
#
# Construct B
data_M = (data_M - mean)/std
#
startNN = time.time()
B_m = np.zeros(shape=(M_m.shape[0], int((M_m.shape[0]-1)/2)+1))
res = model.predict(data_M)

i = 0
j = 0

# Left Boundary
# patch = res[0, :]
# B[i:i+3, j] = patch[:3]
# B[i:i+5, j+1] = patch[3:8]
# B[i+2, j+2] = patch[8]
# #
# # i = i + 2
# # j = j + 1
#
for k in range(0, res.shape[0]):
    patch = res[k]
    # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
    B_m[i + 2, j] = patch[2]

    middle = patch[4:7]
    # middle[0] = np.mean([B[i, j + 1], middle[0]])
    B_m[i+1:i + 4, j + 1] = middle

    B_m[i + 2, j + 2] = patch[8]

    i = i + 2
    j = j + 1
#
# Right Boundary
# patch = res[-1, :]
# # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
# B[i + 2, j] = patch[2]
#
# middle = patch[3:8]
# # middle[0] = np.mean([B[i, j + 1], middle[0]])
# B[i:i+5, j+1] = patch[3:8]
#
# B[i+2:i+5, j+2] = patch[8:]


sum_row_M = M_m.sum(axis=1)
sum_row_B = B_m.sum(axis=1)
diff = sum_row_M - sum_row_B
B_m[0:2, 0] = diff[0:2]
B_m[-2:, -1] = diff[-2:]
#
#
#
# # B[0, 0] = 1
# # B[1,0] = 0.5
# # B[1,1] = 0.5
# # B[-2, -2] = 0.5
# # B[-2, -1] = 0.5
# # B[-1, -1] = 1
#
# Now to Q
B_tmp = B_m
B_m = B_m[np.ix_(reset_m, reset_m_b)]

# row_sums = B_m.sum(axis=1)
# Q_m = B_m / row_sums[:, np.newaxis]
#
# endNN = time.time()
# timeNN = endNN - startNN
B = np.zeros(B_m.shape)
for jj in range(0,B.shape[0]):
    for kk in range(0, B.shape[1]):
        morton = B_m[jj,kk]
        hilbert = B_h[jj,kk]
        if morton == 0 or hilbert == 0:
            B[jj,kk] = morton + hilbert
        else:
            B[jj, kk] = (hilbert + morton)/2

B = B_m + B_h
row_sums = B.sum(axis=1)
Q = B / row_sums[:, np.newaxis]
# Q = Q_h + Q_m

mgr = SemiGeometricMG(A, rhs, Q)
mgr.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
mgr.plot('log')
# plt.savefig('../data/penalty_model_20k.png', dpi=300)
plt.show()
#
# print("NN: ", timeNN)
# print("SG: ", timeSG)
#
# diff = scipy.linalg.norm(Q - L2)/(ne**2)
# # diff = np.average(Q)/np.average(L2)
# print("Distance: ", diff)
