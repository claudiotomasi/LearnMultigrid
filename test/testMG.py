from learn_multigrid.utilities.laplacian import *
from learn_multigrid.mesh.Mesh1D import *
from learn_multigrid.solvers.Jacobi import Jacobi
from learn_multigrid.solvers.GaussSeidel import GaussSeidel
import matplotlib.pyplot as plt
from learn_multigrid.solvers.Multigrid import *
from learn_multigrid.L2_projection.L2Projection import *


def f(x):
    # return np.sin(2 * np.pi * x)
    return np.ones(shape = x.shape)

a = 0
b = 1

# TODO: Be aware of BCs in L2proj
n_el = 20
mesh = Mesh1D(regular = True, ne = n_el)
mesh.construct()
x = mesh.get_mesh()
n_points = mesh.get_np()
# L, X, rhs = laplacian_1d_fd_bc(mesh, mesh.get_mesh(), mesh.get_np(), f)


coarse_n_el = int(n_el/2)
coarse_mesh = Mesh1D(regular = True, ne = coarse_n_el)
coarse_mesh.construct()

# L, X, rhs = laplacian_1d_fd_bc(x, n_points, f)
# print(L.shape)
# L = L.todense()
# mg.solve(smoother = "GaussSeidel", levels=2, max_iterations=20)
# print(mg.get_residual(), mg.get_iterations())

# j = GaussSeidel(L, rhs)
# TODO: stiffness matrix given the mesh!
# TODO: quadrature for rhs (local assembly rhs) - mean value ??
# j.solve(max_iterations=20)

# j2 = Jacobi(L, rhs)


m = Mesh1DRefinement(coarse_ne = 2, n_ref=3)
m.construct()
print(m.get_np())

m_c = Mesh1DRefinement(coarse_ne = 3, n_ref = 2)
m_c.construct()
print(m_c.get_np())
L, X, rhs = laplacian_1d_fd_bc(mesh, f)


# tran_op = L2Projection("pseudo", m, m_c)
#
# Q = tran_op.compute_transfer_1d()
# # print(Q)
# mg = SemiGeometricMG(L, rhs, Q)
# mg.solve(smoother = "GaussSeidel", smooth_steps=1, levels=3, max_iterations=100, error=1e-11)
# mg.plot('linear')
# # j.plot('log')
# # j2.plot('log')
# plt.show()
#
# # start = time.time
# # spsolve(csr_matrix(L), rhs)
# # stop =time.time()

# TODO: check L2_proj normale
# TODO: you're not using stiffness from mesh, just classic lapalcian
