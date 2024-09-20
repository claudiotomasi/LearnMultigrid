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
    # return np.ones(shape = x.shape)*-1
    return -1


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# model = tf.keras.models.load_model('../data/models/new_coarse.h5',  custom_objects={'custom_loss': custom_loss})
model = tf.keras.models.load_model('../data/models/best_penalty_11.h5', compile=False)

# remember, with several levels, if M at a certain point has even dimension
# everything stop. Choose dimension by refinement from even number of element
ne = 1496*10
mesh = Mesh1D(regular = False, ne = ne)
# mesh_C = Mesh1D(regular = True, ne = int(ne/2))
# mesh_C.construct()
coarse = Mesh1D(regular = True, ne = int(ne/2))
mesh.construct()
coarse.x = mesh.get_mesh()[0::2]
coarse.connection_matrix()


q = Quadrature(3)

phi = Function(2)
dphi = Gradient(2)

s = StiffnessMatrix(mesh)
A = s.compute_stiffness_1d(dphi, q)

L, X, rhs = laplacian_1d_fd_bc(mesh, f)

mass = MassMatrix(mesh)
M = mass.compute_mass_1d(phi, q)


# std = np.array([0.00068677, 0.00282781, 0.0007619 , 0.00293974, 0.00073944,
#                 0.00305577, 0.0008198 ])
#
# mean = np.array([0.00012358, 0.00050126, 0.00012705, 0.00050651, 0.0001262 ,
#                  0.00051104, 0.00012932])
std = np.array([0.00051472, 0.00209519, 0.00055687, 0.00216233, 0.0005473 ,
       0.00224975, 0.00060137])
mean = np.array([3.92511629e-05, 1.59982482e-04, 4.07400779e-05, 1.62382236e-04,
       4.04510401e-05, 1.65230747e-04, 4.21643334e-05])

# # prepare NN input
# k = 0
# data_M = np.zeros(shape=(int(mesh.get_ne()/2) - 1, 7))
#
# for i in (range(1, mesh.get_np()-2, 2)):
#     locM = M[i:i+3, i-1:i+4]
#     locM = locM[np.nonzero(locM)]
#     _, idx = np.unique(locM, return_index=True)
#     row = locM[np.sort(idx)]
#     data_M[k, :] = row
#     k = k+1
#
# # Construct B
# data_M = (data_M - mean)/std
#
# startNN = time.time()
# B = np.zeros(shape=(M.shape[0], int((M.shape[0]-1)/2)+1))
# res = model.predict(data_M)
#
# i = 0
# j = 0
#
# # Left Boundary
# # patch = res[0, :]
# # B[i:i+3, j] = patch[:3]
# # B[i:i+5, j+1] = patch[3:8]
# # B[i+2, j+2] = patch[8]
# #
# # i = i + 2
# # j = j + 1
#
# for k in range(0, res.shape[0]):
#     patch = res[k]
#     # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
#     B[i + 2, j] = patch[2]
#
#     middle = patch[4:7]
#     # middle[0] = np.mean([B[i, j + 1], middle[0]])
#     B[i+1:i + 4, j + 1] = middle
#
#     B[i + 2, j + 2] = patch[8]
#
#     i = i + 2
#     j = j + 1
#
# # Right Boundary
# # patch = res[-1, :]
# # # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
# # B[i + 2, j] = patch[2]
# #
# # middle = patch[3:8]
# # # middle[0] = np.mean([B[i, j + 1], middle[0]])
# # B[i:i+5, j+1] = patch[3:8]
# #
# # B[i+2:i+5, j+2] = patch[8:]
#
#
# sum_row_M = M.sum(axis=1)
# sum_row_B = B.sum(axis=1)
# diff = sum_row_M - sum_row_B
# B[0:2, 0] = diff[0:2]
# B[-2:, -1] = diff[-2:]
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
# # Now to Q
# row_sums = B.sum(axis=1)
# Q = B / row_sums[:, np.newaxis]
# endNN = time.time()
# timeNN = endNN - startNN

load = LoadVector(mesh)
rhs = load.compute_rhs_1d(f)
rhs[0] = 0
rhs[-1] = 0

# boundaries:
A[1, 0]= 0
A[-2, -1] = 0
A[0, :] = 0
A[-1, :] = 0
A[0, 0] = 1
A[-1, -1] = 1

# TODO: fare una classe a parte per deep mg
# TODO: diverse network
# TODO: measure time deepMG e SGMG
# cg = CG(A, rhs)
# cg.solve(error=1e-13, max_iterations=300)
# cg.plot('log')
#

# m_c = Q.T @ M @ Q

nmg = NeuralMG(A,rhs,model,M,std,mean)
Q = nmg.transfer_op(M)
# nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
# nmg.plot('log')
nmg = SemiGeometricMG(A ,rhs, Q)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=15)
# plt.plot(nmg.get_track_res(), label="NeuralMG")
# # plt.yscale('log')
# # plt.show()
#
# #
# # startSG = time.time()
# #
# # # TODO: pseudo or quasi ???
tran_op = L2Projection("quasi", mesh, coarse)
#
L2, _ = tran_op.compute_transfer_1d()
mgr = SemiGeometricMG(A ,rhs, L2)
mgr.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=15)
# # mgr.plot('log')
# # plt.savefig('../data/for_HK_1d_more_levels.png', dpi=300)
# # plt.show()
# #
# # print("NN: ", timeNN)
# # print("SG: ", timeSG)
# #
# # diff = scipy.linalg.norm(Q - L2)/(ne**2)
# # # diff = np.average(Q)/np.average(L2)
# # print("Distance: ", diff)
#
# #
plt.close()
plt.rcParams['figure.figsize']=[10, 10]
plt.yscale("log")
# fig, a = plt.subplots(1, 1)
# fig.subplots_adjust(hspace=0.25, wspace=0.25)
# plt.figure()
plt.plot(nmg.get_track_res(),  'o-', lw=1, ms = 5, label='Neural Multigrid')
plt.xlabel('iterations', fontsize=15)
plt.ylabel('residual', fontsize=15)
plt.plot(mgr.get_track_res(), 's-', lw=1, ms = 5, label='SG Multigrid',)
plt.grid(True, linestyle='--', color='k', alpha=0.3, lw = 1)
plt.title("Residual decreasing", fontsize = 28, pad = 10)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=20)
plt.savefig('./plot.png', dpi=300)
plt.show()
nmg_res = nmg.get_track_res()
smg_res = mgr.get_track_res()

np.save("./res_NMG", nmg_res)
np.save("./res_SMG", smg_res)