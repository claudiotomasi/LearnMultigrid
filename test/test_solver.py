import numpy as np
from learn_multigrid.solvers.Solver import *
from learn_multigrid.solvers.Jacobi import Jacobi
from learn_multigrid.solvers.GaussSeidel import GaussSeidel
import time
from scipy.sparse import diags

A = np.random.rand(3, 3)


rhs = np.random.rand(3, 1)
# print(rhs)

sol = DirectSolver(A, rhs)
# print(sol.get_matrix())

sol.solve()

# print(sol.get_solution())

solution = sol.get_solution()

# print(sol.get_residual_vector())
# print(sol.get_residual())


start = time.time()


A = csc_matrix([[30, 1, 15], [28, 60, 3], [100, 19, 150]])
sol = DirectSolver(A, rhs)
# print(sol.get_matrix())

sol.solve()
print(sol.get_residual())
jac = Jacobi(A, rhs)


jac.solve()
end = time.time()
# print(end - start)
print(jac.get_residual())
print(jac.get_iterations())

gauss = GaussSeidel(A, rhs)
gauss.solve()
print(gauss.get_residual())
print(gauss.get_iterations())
