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

model = tf.keras.models.load_model('../data/models/to19k_Column.h5')
std = np.array([0.00049838, 0.00202781, 0.00051752, 0.00208096, 0.00052413])
mean = np.array([6.83689748e-05, 2.75354520e-04, 6.93082852e-05, 2.77661475e-04, 6.95224521e-05])

column_names = ["ne", "iterations", "residual", "last_residual"]
df = pd.DataFrame()

iterations = []
elements = []
number = 0
for ne in range(200, 30001, 200):
    elements.append(ne)
    number = number + 1
    mesh = Mesh1D(regular = False, ne = ne)
    mesh.construct()

    q = Quadrature(3)

    phi = Function(2)
    dphi = Gradient(2)

    s = StiffnessMatrix(mesh)
    A = s.compute_stiffness_1d(dphi, q)
    # boundaries:
    A[1,0]= 0
    A[-2, -1] = 0
    A[0,:] = 0
    A[-1,:] = 0
    A[0,0] = 1
    A[-1,-1] = 1

    L, X, rhs = laplacian_1d_fd_bc(mesh, f)

    mass = MassMatrix(mesh)
    M = mass.compute_mass_1d(phi, q)

    nmg = NeuralMG(A,rhs, model, M, std, mean)
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

    df.to_csv('./ne_vs_iterations.csv')

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
plt.savefig('../data/plots/ne_vs_iterations.png', dpi=300)
plt.close('all')
