import numpy as np
from learn_multigrid.solvers.Solver import *
from learn_multigrid.solvers.CG import CG
from learn_multigrid.solvers.Jacobi import Jacobi

from learn_multigrid.solvers.GaussSeidel import GaussSeidel
import time
from scipy.sparse import diags

A = np.random.rand(3, 3)



# print(rhs)


A = csc_matrix([[30, 1, 15], [28, 60, 3], [100, 19, 150]])
rhs = np.array([1, 2, 3]).reshape((3,1))
sol = CG(A, rhs)

sol.solve(max_iterations=100)
print(sol.get_solution())
print(sol.get_residual)
print(sol.get_iterations)

sol.plot('linear')
plt.show()

