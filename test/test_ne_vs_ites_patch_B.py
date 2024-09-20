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
import pandas as pd

import numpy as np
# from sklearn.model_selection import train_test_split

from tensorflow import keras
# from tensorflow.keras import layers

import matplotlib.pyplot as plt
from numpy.random import seed
seed(40)
tf.random.set_seed(41)

def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('../data/models/best_penalty_2.h5',  compile=False)
std = np.array([0.0004888 , 0.00201229, 0.00054178, 0.00209122, 0.00052595,
                0.00217302, 0.00058262])

mean = np.array([6.75344883e-05, 2.73653317e-04, 6.92921701e-05, 2.76313787e-04,
                 6.88647231e-05, 2.78579309e-04, 7.04249314e-05])

column_names = ["ne", "iterations", "residual", "last_residual"]
df = pd.DataFrame()

iterations = []
elements = []
number = 0
for ne in range(200, 35000, 500):
    elements.append(ne)
    number = number + 1
    mesh = Mesh1D(regular = False, ne = ne)
    mesh.construct()

    q = Quadrature(3)

    phi = Function(2)
    dphi = Gradient(2)

    s = StiffnessMatrix(mesh)
    A = s.compute_stiffness_1d(dphi, q)


    # L, X, rhs = laplacian_1d_fd_bc(mesh, f)

    mass = MassMatrix(mesh)
    M = mass.compute_mass_1d(phi, q)

    # prepare NN input
    k = 0
    data_M = np.zeros(shape=(int(mesh.get_ne() / 2) - 1, 7))

    for i in (range(1, mesh.get_np() - 2, 2)):
        locM = M[i:i + 3, i - 1:i + 4]
        locM = locM[np.nonzero(locM)]
        _, idx = np.unique(locM, return_index=True)
        row = locM[np.sort(idx)]
        data_M[k, :] = row
        k = k + 1

    # Construct B
    data_M = (data_M - mean) / std

    B = np.zeros(shape=(M.shape[0], int((M.shape[0] - 1) / 2) + 1))
    res = model.predict(data_M)

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

    sum_row_M = M.sum(axis=1)
    sum_row_B = B.sum(axis=1)
    diff = sum_row_M - sum_row_B
    B[0:2, 0] = diff[0:2]
    B[-2:, -1] = diff[-2:]

    # Now to Q
    row_sums = B.sum(axis=1)
    Q = B / row_sums[:, np.newaxis]

    load = LoadVector(mesh)
    rhs = load.compute_rhs_1d(f)
    rhs[0] = 0
    rhs[-1] = 0

    # boundaries:
    A[1, 0] = 0
    A[-2, -1] = 0
    A[0, :] = 0
    A[-1, :] = 0
    A[0, 0] = 1
    A[-1, -1] = 1

    nmg = SemiGeometricMG(A, rhs, Q)
    nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
    it = nmg.get_iterations()
    iterations.append(it)
    res = nmg.get_track_res()
    last_res = nmg.get_residual()
    del nmg

    new_df = pd.DataFrame({'ne': ne,
                           'iterations': it,
                           'residuals': [np.transpose(res)],
                           'last_residual': last_res}, index=[number])
    df = df.append(new_df)
    # plt.savefig('../data/19k_model_15000.png', dpi=300)
    plt.show()

    df.to_csv('./ne_vs_iterations_patch_B.csv')

plt.figure()
plt.xlabel('N_Elements')
plt.ylabel('N_Iterations')
plt.title("ne vs iterations")
plt.plot(elements, iterations, label="Iterations behaviour")
_, top = plt.ylim()
plt.ylim(3,  top+4)
# plt.yscale('log')

plt.legend()
# plt.show()
plt.savefig('../data/plots/ne_vs_iterations_bigger_patch.png', dpi=300)
plt.close('all')
