import numpy as np
from learn_multigrid.assembly.LoadVector import *
from scipy.sparse import spdiags


# compute laplacian 1d in finite difference
# N in input are the number of points(from x0 to xN)
# thus, (N-1) is n_el
def laplacian_1d_fd(x, N, f):
    h = (x[-1] - x[0])/(N-1)
    X = x[1:N - 1]
    X = X.reshape((N-2, 1))
    rhs = f(X)

    data1 = [2]*(N-2)
    data2 = [-1]*(N-2)
    L = spdiags([ data2, data1, data2 ], [-1, 0, 1], (N-2), (N-2))
    L = (1 / h ** 2) * L
    return L, X, rhs


def laplacian_1d_fd_bc(m, f):
    x = m.get_mesh()
    N = m.get_np()
    h = (x[-1] - x[0])/(N-1)
    # X = x[1:N - 1]
    X = x
    # X = X.reshape((N-2, 1))
    X = X.reshape((N, 1))
    # rhs = f(X)*(h**2)
    # rhs[0] = 0
    # rhs[-1] = 0

    load = LoadVector(m)
    rhs = load.compute_rhs_1d(f)
    rhs[0] = 0
    rhs[-1] = 0

    data1 = [2]*(N)
    data2 = [-1]*(N)
    # L = spdiags([ data2, data1, data2 ], [-1, 0, 1], (N-2), (N-2))

    L = spdiags([data2, data1, data2], [-1, 0, 1], (N), (N))
    L = L.tolil()
    L = (1 / h ** 2) * L
    # boundaries:
    L[1,0]= 0
    L[-2, -1] = 0
    L[0,:] = 0
    L[-1,:] = 0
    L[0,0] = 1
    L[-1,-1] = 1


    # For symmetry:
    # L[1, 0] = 0
    # L[-2, -1] = 0

    return L, X, rhs
