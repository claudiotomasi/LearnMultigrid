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


def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)


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


def solve(nn,stiff,b, mass_mat, levels, error = 1e-10):
    sol = np.zeros(shape=(stiff.shape[0], 1))
    it = 0
    track_res = np.ndarray(shape=(0, 1), dtype=float)
    for i in range(0, 20):
        it += 1
        print("It: ", it)
        residual_vector = b - stiff.dot(sol)
        residual = np.linalg.norm(residual_vector)
        if it <= 1:
            residual_vector = np.ones(shape=sol.shape)
            residual = np.linalg.norm(residual_vector)
        track_res = np.vstack((track_res, residual))
        print(residual)
        if residual <= error:
            # print("Reached convergence level: ", levels)
            break
        # self.solution = self.v_cycle(A, self.solution, self.rhs, smoother, levels)
        sol = vcycle(nn, stiff, mass_mat, sol, b, 3, error, levels)
    return track_res

def vcycle(nn, stiff, mass_mat, u0, b, smooth_steps, error, levels):
    print(levels)
    levels -= 1
    s = GaussSeidel(stiff, b)
    # Pre - Smoothing
    # print("pre-smoothing")
    # print(A, u0)
    s.solve(max_iterations=smooth_steps, initial_guess=u0, error = error)
    u = s.get_solution()
    res = b - stiff.dot(u)

    dimension = mass_mat.shape[0]
    data = prepare_input(mass_mat)
    transfer = compute_B(dimension, nn, data)
    M_sum = mass_mat.sum(axis=1)
    B_sum = transfer.sum(axis=1)
    MB_diff = M_sum - B_sum
    # print("MB_diff: ", MB_diff[0:2].shape, "transfer[0:2, 0]: ",transfer[0:2, 0].shape )
    transfer[0:2, 0] = MB_diff[0:2]
    transfer[-2:, -1] = MB_diff[-2:]

    # Now to Q
    B_sums = transfer.sum(axis=1)
    i = transfer / B_sums[:, np.newaxis]
    # res_coarse = np.dot(i.T, res)
    res_coarse = i.T @ res
    # A = A.toarray()
    # A_coarse = np.dot(i.T, A)
    # A_coarse = np.linalg.multi_dot([i.T, A, i])
    # TODO: It is possibile that with more levels, A and M coarse cannot be csr since we need them for interpolator
    # TODO: possibile solution -> prepare first the interpolators saving them in cache, then move to solve  phase
    A_coarse = i.T @ stiff @ i
    A_coarse = csr_matrix(A_coarse)

    # M_coarse = np.linalg.multi_dot([i.T, M, i])
    M_coarse = i.T @ mass_mat @ i
    M_coarse = csr_matrix(M_coarse)
    M_coarse = M_coarse.toarray()
    # A_coarse = np.dot(A_coarse, i)
    # print(A_coarse)
    if levels != 1:
        u_coarse = vcycle(nn, A_coarse, M_coarse, np.zeros(shape=(A_coarse.shape[0], 1)),
                          res_coarse, 3, error, levels)
    else:
        print("Direct Solving")
        u_coarse = np.reshape(spsolve(A_coarse, res_coarse, use_umfpack = False), newshape=(A_coarse.shape[0],1))
    # Correction
    # u = u + np.dot(i, u_coarse)
    u = u + i @ u_coarse

    # Post - Smoothing
    # print("post-smoothing")
    s.solve(max_iterations=smooth_steps, initial_guess=u, error = error)
    u = s.get_solution()

    return u


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('../data/models/best_penalty_2.h5', compile=False)

ne = 5120
mesh = Mesh1D(regular = False, ne = ne)
# mesh_C = Mesh1D(regular = True, ne = int(ne/2))
# mesh_C.construct()
coarse = Mesh1D(regular = True, ne = int(ne/2))
mesh.construct()
coarse.x = mesh.get_mesh()[0::2]
coarse.connection_matrix()


q = Quadrature(3)
q = Quadrature(3)

phi = Function(2)
dphi = Gradient(2)

s = StiffnessMatrix(mesh)
A = s.compute_stiffness_1d(dphi, q)

load = LoadVector(mesh)
rhs = load.compute_rhs_1d(f)
rhs[0] = 0
rhs[-1] = 0

# L, X, rhs = laplacian_1d_fd_bc(mesh, f)

mass_ = MassMatrix(mesh)
M = mass_.compute_mass_1d(phi, q)

# dim = M.shape[0]
# data_M = prepare_input(M)
# B = compute_B(dim, model, data_M)
#
#
# sum_row_M = M.sum(axis=1)
# sum_row_B = B.sum(axis=1)
# diff = sum_row_M - sum_row_B
# B[0:2, 0] = diff[0:2]
# B[-2:, -1] = diff[-2:]
#
# # Now to Q
# row_sums = B.sum(axis=1)
# Q = B / row_sums[:, np.newaxis]



# boundaries:
A[1, 0]= 0
A[-2, -1] = 0
A[0, :] = 0
A[-1, :] = 0
A[0, 0] = 1
A[-1, -1] = 1

residuals = solve(model, A, rhs, M, 5, error = 1e-10)

plt.plot(residuals, label="NeuralMG")
plt.yscale('log')
plt.legend()
plt.title("Residual decreasing")
plt.ylabel('residual')
plt.xlabel('iterations')
# plt.show()

# TODO: fare una classe a parte per deep mg
# TODO: diverse network
# TODO: measure time deepMG e SGMG
# cg = CG(A, rhs)
# cg.solve(error=1e-13, max_iterations=300)
# cg.plot('log')
#

# m_c = Q.T @ M @ Q

# nmg = SemiGeometricMG(A,rhs,Q)
# nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
# # nmg.plot('log')
# plt.plot(nmg.get_track_res(), label="NeuralMG")
# plt.yscale('log')
#
# #
startSG = time.time()

# TODO: pseudo or quasi ???
tran_op = L2Projection("quasi", mesh, coarse)

L2, _ = tran_op.compute_transfer_1d()
m_c = L2.T @ M @ L2
endSG = time.time()
timeSG = endSG - startSG
mgr = SemiGeometricMG(A,rhs, L2)
mgr.solve(levels=5, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
mgr.plot('log')
# plt.savefig('../data/19k_model_15000.png', dpi=300)
plt.show()
#
# print("NN: ", timeNN)
# print("SG: ", timeSG)
#
# diff = scipy.linalg.norm(Q - L2)/(ne**2)
# # diff = np.average(Q)/np.average(L2)
# print("Distance: ", diff)
