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
import pickle

import numpy as np
# from sklearn.model_selection import train_test_split

from tensorflow import keras
# from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow.keras.losses
import tensorflow.keras.backend as K



def f(x):
    # return np.sin(2 * np.pi * x)
    # return np.ones(shape = x.shape)*-1
    return -1


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# model = tf.keras.models.load_model('../data/models/new_coarse.h5',  custom_objects={'custom_loss': custom_loss})
model = tf.keras.models.load_model('../data/models/best_penalty_11.h5', compile=False)

times_NMG = []
times_SGMG = []
dim = []
for kk in range(1, 11):
    ne = 1024 * 10 * kk
    print("ne: ", ne)
    mesh = Mesh1D(regular = False, ne = ne)
    dim.append(mesh.get_np())
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


    std = np.array([0.00051472, 0.00209519, 0.00055687, 0.00216233, 0.0005473 ,
           0.00224975, 0.00060137])
    mean = np.array([3.92511629e-05, 1.59982482e-04, 4.07400779e-05, 1.62382236e-04,
           4.04510401e-05, 1.65230747e-04, 4.21643334e-05])

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

    nmg = NeuralMG(A,rhs,model,M,std,mean)
    st_N = time.time()
    Q = nmg.transfer_op(M)
    time_Q = time.time() - st_N
    nmg = SemiGeometricMG(A ,rhs, Q)
    times_NMG.append(time_Q)

    st_SGMG = time.time()
    tran_op = L2Projection("quasi", mesh, coarse)
    L2, _ = tran_op.compute_transfer_1d()
    time_L2 = time.time() - st_SGMG
    times_SGMG.append(time_L2)

with open("time_NMG.txt", "wb") as fp:   #Pickling
    pickle.dump(times_NMG, fp)
with open("time_SGMG.txt", "wb") as fp:   #Pickling
    pickle.dump(times_SGMG, fp)

plt.close()
plt.rcParams['figure.figsize']=[10, 10]
plt.yscale("log")
# fig, a = plt.subplots(1, 1)
# fig.subplots_adjust(hspace=0.25, wspace=0.25)
# plt.figure()
plt.plot(dim, times_NMG,  'o-', lw=1, ms = 5, label='Neural Multigrid')
plt.xlabel('dimension', fontsize=26)
plt.ylabel('time', fontsize=26)
plt.plot(dim,times_SGMG , 's-', lw=1, ms = 5, label='SemiGM Multigrid',)
plt.grid(True, linestyle='--', color='k', alpha=0.3, lw = 1)
plt.title("Residual decreasing", fontsize = 35, pad = 12)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=20)
# plt.savefig('../data/conv_HK.png', dpi=300)
plt.show()
