import pyamg
import scipy.io
from scipy.sparse import lil_matrix
import time
import numpy as np
from learn_multigrid.solvers.GaussSeidel import *
from pyamg.relaxation.relaxation import gauss_seidel
from pyamg.util.linalg import norm
from scipy.sparse.linalg import splu




path_matlab = "../data/from_matlab/"
problem = path_matlab + "smile50.mat"
pp = scipy.io.loadmat(problem)
A = lil_matrix(pp['A'])
M = lil_matrix(pp['M'])
rhs = pp['rhs']

# path_data = "/Users/claudio/Desktop/PhD/Tesi/dati_test/smile50/"
ml = pyamg.ruge_stuben_solver(A, max_levels=2, presmoother=['gauss_seidel', {'iterations': 3}], coarse_solver='splu')
b = rhs.flatten()
res = []
st = time.time()
x = ml.solve(b, tol=1e-09, residuals = res, cycle = 'V')
t = time.time()-st
print("Time: ", t)
print("residual: ", np.linalg.norm(b-A*x))
# plt.close()
# plt.rcParams['figure.figsize']=[10, 9]
# plt.yscale("log")
# plt.plot(nn_res,  'o-', lw=1, ms = 5, label='Neural Multigrid old')
# plt.xlabel('iterations', fontsize=15)
# plt.ylabel('residual', fontsize=15)
# plt.grid(True, linestyle='--', color='k', alpha=0.3, lw = 1)
# plt.title("Residual decreasing", fontsize = 28, pad = 10)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.legend(fontsize=20)
# # plt.savefig('../data/conv_HK.png', dpi=300)
# plt.show()
# 
# 
#

gs = GaussSeidel(A,rhs)
st = time.time()
# gs.solve(1)
t_gs = time.time()-st
print(t_gs)
gs_sol = gs.get_solution()

# x0 = np.zeros((A.shape[0],1))
# st = time.time()
# gauss_seidel(A, x0, rhs, iterations=1)
# t_pyamg = time.time()-st
# print(t_pyamg)
# print(norm(rhs-A*x0))
# sol_pyamg = x0





