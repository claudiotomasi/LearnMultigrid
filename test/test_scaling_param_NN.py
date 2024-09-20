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
import joblib
from sklearn.preprocessing import MinMaxScaler

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


def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('../data/models/to10k_Column.h5')
scaling_model = tf.keras.models.load_model('../data/models/scaling_param.h5')
ne = 3000
mesh = Mesh1D(regular = False, ne = ne)
mesh_C = Mesh1D(regular = True, ne = int(ne/2))
mesh.construct()
mesh_C.construct()

scale_x = joblib.load("../data/models/scaler_x.save")
scale_y = joblib.load("../data/models/scaler_y.save")

q = Quadrature(3)

phi = Function(2)
dphi = Gradient(2)

s = StiffnessMatrix(mesh)
A = s.compute_stiffness_1d(dphi, q)

L, X, rhs = laplacian_1d_fd_bc(mesh, f)

mass = MassMatrix(mesh)
M = mass.compute_mass_1d(phi, q)

M_ref = np.array([5.5727e-05,  2.1157e-04,  5.0056e-05, 2.0113e-04, 5.0509e-05])

B_ref = np.array([1.3895e-05,  1.2746e-04,  2.4752e-04, 1.8083e-04, 4.5703e-05, 3.5024e-07])

std = np.array([17.84240136, 16.79429439, 16.00760005, 17.01769213, 18.20427564])
mean = np.array([3.88546571, 3.54329983, 3.26208686, 3.47256276, 3.71214393])

# prepare NN input
k = 0
data_M = np.zeros(shape=(int(mesh.get_ne()/2) - 1, 5))
scale_M = np.zeros(shape=(int(mesh.get_ne()/2) - 1, 5))

a = M[0, np.nonzero(M[0, :])]
b = M[1, np.nonzero(M[1, :])]
c = M[2, np.nonzero(M[2, :])]
row = np.concatenate((a, b, c), axis = 1)
row = np.unique(row)

# data_M[1, :] = np.concatenate(([0], row))
for i in (range(1, mesh.get_np()-2, 2)):
    locM = M[i:i+3, i-1:i+2]
    locM = locM[np.nonzero(locM)]
    _, idx = np.unique(locM, return_index=True)
    row = locM[np.sort(idx)]
    data_M[k, :] = row / M_ref
    k = k+1

scale_M = (data_M - mean)/std



startNN = time.time()

B = np.zeros(shape=(M.shape[0], int(M.shape[0]/2)+1))

res_scaled = scaling_model.predict(scale_M)
B_res = res_scaled * B_ref

i = 0
k = 0
for j in range(1, B.shape[1]-1):
    column = B_res[k]
    # column = np.array([column])
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
# row = data_M[1, :]
# res = model.predict(np.array([row]))
# B[1, 0:3] = res[0]
# B = scale_y.inverse_transform(B)
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
nmg = NeuralMG(A,rhs, model, M, std, mean)
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
# plt.savefig('../data/dropout_ne20000.png', dpi=300)
plt.show()

print("NN: ", timeNN)
print("SG: ", timeSG)

diff = scipy.linalg.norm(Q - L2)/(ne**2)
# diff = np.average(Q)/np.average(L2)
print("Distance: ", diff)
