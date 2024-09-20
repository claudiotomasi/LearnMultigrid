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


def f(x):
    return np.sin(2 * np.pi * x)
    # return np.ones(shape = x.shape)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_middle = tf.keras.models.load_model('../data/model_middle.h5')
model_with_b = tf.keras.models.load_model('../data/model_with_b.h5')


mesh = Mesh1D(regular = False, ne = 250)
mesh_C = Mesh1D(regular = False, ne = 125)
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

# a = M[1, np.nonzero(M[1, :])]
# b = M[2, np.nonzero(M[2, :])]
# c = M[3, np.nonzero(M[3, :])]
# row = np.concatenate((a, b, c), axis = 1)
# row = np.unique(row)

std = np.array([0.00329779, 0.01337664, 0.0034045 , 0.01360203, 0.00340482,
                0.01384641, 0.00353129])
mean = np.array([0.00183888, 0.00741199, 0.00186712, 0.00746546, 0.00186562,
                 0.00751872, 0.00189374])
# row = (row - mean)/std

# res = model.predict(np.array([row]))
#
#
# inter = Intersection(mesh, mesh_C)
# inter.find_intersections1d()
# intersections, int_coord, _ = inter.get_info()
#
# q = Quadrature(3)
#
# phi = Function(2)
#
# coup_op = CouplingOperator(inter, mesh, mesh_C)
#
# B = coup_op.compute_b_1d(q, phi)

new_std = np.array([0.00329597, 0.01337274, 0.00340424, 0.01359644, 0.00340257, 0.01385977, 0.00354047, 0.01504948])
new_mean = np.array([0.00182711, 0.00736647, 0.00185612, 0.00742591, 0.00185683, 0.00747108, 0.0018787 , 0.00791887])

# TODO: be aware, B constructed with that loop, only works for coarse_ne = ne/2s


data_M = np.zeros(shape=(mesh.get_np()-4, 7))

a = M[0, np.nonzero(M[0, :])]
b = M[1, np.nonzero(M[1, :])]
c = M[2, np.nonzero(M[2, :])]
row = np.concatenate((a, b, c), axis = 1)
row = np.unique(row)
sum_row_M = M.sum(axis=1)
sum_row_M = sum_row_M[2:-2]

# data_M[1, :] = np.concatenate(([0], row))
k = 0
for i in range(2, mesh.get_np() - 2):
    a = M[i-1, np.nonzero(M[i-1, :])]
    b = M[i, np.nonzero(M[i, :])]
    c = M[i+1, np.nonzero(M[i+1, :])]
    row = np.concatenate((a, b, c), axis = 1)[0]
    _, idx = np.unique(row, return_index=True)
    row = row[np.sort(idx)]
    row = np.unique(row)
    data_M[k, :] = row
    k = k + 1

a = M[-3, np.nonzero(M[-3, :])]
b = M[-2, np.nonzero(M[-2, :])]
c = M[-1, np.nonzero(M[-1, :])]
row = np.concatenate((a, b, c), axis = 1)
row = np.unique(row)
# data_M[-2, :] = np.concatenate((row, [0]))


data_M_normed = (data_M - mean)/std
startNN = time.time()

# First network
middle = model_middle.predict(data_M_normed)

new_data = np.concatenate((data_M, middle), axis = 1)

new_data_normed = (new_data - new_mean)/new_std

# Construct B
B = np.zeros(shape=(mesh.get_np(), mesh_C.get_np()))
res = model_with_b.predict(new_data_normed)

res = np.insert(res, [1], middle, axis=1)

k = 0
j = 0
count = 0
for i in range(2, mesh.get_np() - 2):
    row = res[k]

    # diff = sum_row_M[i] - sum(row)
    # if row[2] < 0:
    #     sum_ex = row[0] + row[1]
    #     row[2] = sum_row_M[k] - sum_ex
    B[i, j:j + 3] = row
    # print(res, i, j)
    count += 1
    if count == 2:
        count = 0
        j += 1
    print(row)
    k = k + 1
# i += 1
# row = data_M[i, :]
# res = model.predict(np.array([row]))
# sum_row_B = B.sum(axis=1)


# row = data_M[1, :]
# res = model.predict(np.array([row]))
# B[1, 0:3] = res[0]
B[0, 0] = 1
B[1,0] = 0.5
B[1,1] = 0.5
B[-2, -2] = 0.5
B[-2, -1] = 0.5
B[-1, -1] = 1

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
nmg = NeuralMG(A,rhs, Q)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
nmg.plot('log')

startSG = time.time()

# TODO: pseudo or quasi ???
tran_op = L2Projection("quasi", mesh, mesh_C)

L = tran_op.compute_transfer_1d()
endSG = time.time()
timeSG = endSG - startSG
mgr = SemiGeometricMG(A ,rhs, L)
mgr.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
mgr.plot('log')
# plt.savefig('../data/itXres.png', dpi=300)
plt.show()

print("NN: ", timeNN)
print("SG: ", timeSG)
