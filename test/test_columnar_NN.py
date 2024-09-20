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
model = tf.keras.models.load_model('../data/models/best_new_coarse_5.h5', compile=False)

ne = 10000
mesh = Mesh1D(regular = False, ne = ne)
mesh_C = Mesh1D(regular = True, ne = int(ne/2))
mesh.construct()
mesh_C.construct()


q = Quadrature(3)

phi = Function(2)
dphi = Gradient(2)

s = StiffnessMatrix(mesh)
A = s.compute_stiffness_1d(dphi, q)

L, X, rhs = laplacian_1d_fd_bc(mesh, f)

mass = MassMatrix(mesh)
M = mass.compute_mass_1d(phi, q)


std = np.array([0.00068677, 0.00282781, 0.0007619 , 0.00293974, 0.00073944,
                0.00305577, 0.0008198 ])

mean = np.array([0.00012358, 0.00050126, 0.00012705, 0.00050651, 0.0001262 ,
                 0.00051104, 0.00012932])


# prepare NN input
k = 0
data_M = np.zeros(shape=(int(mesh.get_ne()/2) - 1, 7))

a = M[0, np.nonzero(M[0, :])]
b = M[1, np.nonzero(M[1, :])]
c = M[2, np.nonzero(M[2, :])]
row = np.concatenate((a, b, c), axis = 1)
row = np.unique(row)

# data_M[1, :] = np.concatenate(([0], row))
for i in (range(1, mesh.get_np()-2, 2)):
    locM = M[i:i+3, i-1:i+4]
    locM = locM[np.nonzero(locM)]
    _, idx = np.unique(locM, return_index=True)
    row = locM[np.sort(idx)]
    data_M[k, :] = row
    k = k+1

a = M[-3, np.nonzero(M[-3, :])]
b = M[-2, np.nonzero(M[-2, :])]
c = M[-1, np.nonzero(M[-1, :])]
row = np.concatenate((a, b, c), axis = 1)
row = np.unique(row)
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
B = np.zeros(shape=(M.shape[0], int(M.shape[0]/2)+1))
res = model.predict(data_M)

i = 0
k = 0
for j in range(1, B.shape[1]-1):
    column = res[k]
    # if j == B.shape[1]-2:
    #     column = column[1:]
    #     B[i:i + 5, j] = column
    # else:
    B[i:i+5, j] = column
    i = i + 2
    k = k + 1

# i += 1
# row = data_M[i, :]
# res = model.predict(np.array([row]))
# row = data_M[1, :]
# res = model.predict(np.array([row]))
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
# rhs[-1] = 0
x = mesh.get_mesh()

# boundaries:
A[1,0]= 0
A[-2, -1] = 0
A[0,:] = 0
A[-1,:] = 0
A[0,0] = 1
A[-1,-1] = 1

# TODO: fare una classe a parte per deep mg
# TODO: diverse network
# TODO: measure time deepMG e SGMG
# cg = CG(A, rhs)
# cg.solve(error=1e-13, max_iterations=300)
# cg.plot('log')
#
nmg = SemiGeometricMG(A,rhs,Q)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
nmg.plot('log')

startSG = time.time()

# TODO: pseudo or quasi ???
tran_op = L2Projection("quasi", mesh, mesh_C)

L2, _ = tran_op.compute_transfer_1d()
endSG = time.time()
timeSG = endSG - startSG
mgr = SemiGeometricMG(A ,rhs, L2)
mgr.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
mgr.plot('log')
# plt.savefig('../data/19k_model_15000.png', dpi=300)
plt.show()

print("NN: ", timeNN)
print("SG: ", timeSG)

diff = scipy.linalg.norm(Q - L2)/(ne**2)
# diff = np.average(Q)/np.average(L2)
print("Distance: ", diff)
