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
import matplotlib.pyplot as plt


def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('../data/Columnmodel_2_2.h5')
ne = 500
mesh = Mesh1D(regular = False, ne = ne)
mesh.construct()


q = Quadrature(3)

phi = Function(2)
dphi = Gradient(2)

s = StiffnessMatrix(mesh)
A = s.compute_stiffness_1d(dphi, q)

L, X, rhs = laplacian_1d_fd_bc(mesh, f)

mass = MassMatrix(mesh)
M = mass.compute_mass_1d(phi, q)


std = np.array([0.00089083, 0.00364883, 0.00093721, 0.00374349, 0.00093641])
mean = np.array([0.00019376, 0.00078147, 0.00019697, 0.00078759, 0.00019682])

A[1,0]= 0
A[-2, -1] = 0
A[0,:] = 0
A[-1,:] = 0
A[0,0] = 1
A[-1,-1] = 1


nmg = NeuralMG(A,rhs, model, M, std, mean)
nmg.solve(levels = 3, smoother="GaussSeidel", smooth_steps=3, error=1e-10, max_iterations=40)
nmg.plot('log')

plt.show()

