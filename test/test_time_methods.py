from __future__ import absolute_import, division, print_function, unicode_literals
from learn_multigrid.solvers.Multigrid import *
from learn_multigrid.L2_projection.L2Projection import *
from learn_multigrid.L2_projection.CouplingOperator import *
from learn_multigrid.assembly.StiffnessMatrix import *
from learn_multigrid.utilities.laplacian import *

import time
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('../data/models/best_penalty_11.h5', compile=False)
std = np.array([0.00051472, 0.00209519, 0.00055687, 0.00216233, 0.0005473 ,
       0.00224975, 0.00060137])
mean = np.array([3.92511629e-05, 1.59982482e-04, 4.07400779e-05, 1.62382236e-04,
       4.04510401e-05, 1.65230747e-04, 4.21643334e-05])

track_time_NN = np.ndarray(shape=(0, 1), dtype=float)
track_time_SG = np.ndarray(shape=(0, 1), dtype=float)
track_ne = np.ndarray(shape=(0, 1), dtype=float)
ne = 100
for i in range(10):
    coarse_ne = int(ne/2)
    mesh = Mesh1D(regular = False, ne = ne)
    mesh.construct()
    mesh_C = Mesh1D(regular=True, ne=coarse_ne)
    mesh_C.x = mesh.get_mesh()[0::2]
    mesh_C.connection_matrix()

    q = Quadrature(3)
    phi = Function(2)
    dphi = Gradient(2)
    mass = MassMatrix(mesh)
    M = mass.compute_mass_1d(phi, q)
    s = StiffnessMatrix(mesh)
    A = s.compute_stiffness_1d(dphi, q)
    startNN = time.time()

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
    nmg = NeuralMG(A, rhs, model, M, std, mean)
    nmg.transfer_op(M)
    endNN = time.time()
    timeNN = endNN - startNN

    startSG = time.time()

    tran_op = L2Projection("quasi", mesh, mesh_C)

    L2, timeIntersections = tran_op.compute_transfer_1d()
    endSG = time.time()

    timeSG = endSG - startSG

    print("NN: ", timeNN)
    print("SG: ", timeSG)

    track_time_NN = np.vstack((track_time_NN, timeNN))
    track_time_SG = np.vstack((track_time_SG, timeSG))
    track_ne = np.vstack((track_ne, ne))

    ne = ne + 1000

plt.plot(track_ne, track_time_NN, label='NN')
plt.plot(track_ne, track_time_SG, label='SG')
plt.yscale('linear')
plt.legend()
plt.title("Time L2-proj")
plt.ylabel('Time (s)')
plt.xlabel('Dimension (ndof)')
plt.savefig('../data/time_plot2_linear_2.png', dpi=300)
plt.show()


# TODO : added in L2Projection, the return of the time spent in intersections
# So that I could remove it from the total time of computation of L2-proj
