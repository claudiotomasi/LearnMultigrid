from learn_multigrid.utilities.laplacian import  *
from learn_multigrid.mesh.Mesh1D import *
from learn_multigrid.solvers.Jacobi import Jacobi
import matplotlib.pyplot as plt
from learn_multigrid.solvers.GaussSeidel import GaussSeidel



def f(x):
    return np.cos(x)


a = 0
b = 1

mesh = Mesh1D(regular = True, ne = 6)
mesh.construct()
x = mesh.get_mesh()
n_points = mesh.get_np()

L, X_no_bound, rhs = laplacian_1d_fd(x, n_points, f)

# print(L)

j = Jacobi(L, rhs)

j.solve()
# print(j.get_iterations(), j.get_residual())
res = j.get_track_res()

g = GaussSeidel(L, rhs)
g.solve()

g_res = g.get_track_res()
# print(res)
g.plot('log')
j.plot('log')

plt.show()

