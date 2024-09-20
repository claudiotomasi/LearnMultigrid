from learn_multigrid.L2_projection.L2Projection import *
from learn_multigrid.assembly.MassMatrix import MassMatrix
from learn_multigrid.mesh.Mesh1D import *
from learn_multigrid.L2_projection.Intersection import *
# from learn_multigrid.L2_projection.CouplingOperator import *
from learn_multigrid.utilities.plots import *

# test to perform l2proj with object class

n_el = 12
fine_mesh = Mesh1D(regular = False, ne = n_el)
fine_mesh.construct()

coarse_n_el = int(n_el/2)
coarse_mesh = Mesh1D(regular = True, ne = coarse_n_el)
coarse_mesh.construct()

tran_op = L2Projection("quasi", fine_mesh, coarse_mesh)

L = tran_op.compute_transfer_1d()

print(L)
