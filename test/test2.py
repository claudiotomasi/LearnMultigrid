import sys
import os

from learn_multigrid.assembly.MassMatrix import MassMatrix

# sys.path.append('../')
from learn_multigrid.L2_projection.Intersection import *
from learn_multigrid.L2_projection.CouplingOperator import *
from learn_multigrid.utilities.plots import *


fine_mesh = Mesh1D(regular = True, ne = 4)
fine_mesh.construct()
fine_x = fine_mesh.get_mesh()

coarse_mesh = Mesh1D(regular = True, ne = 2)
coarse_mesh.construct()
coarse_x = coarse_mesh.get_mesh()

inter = Intersection(fine_mesh, coarse_mesh)
inter.find_intersections1d()
intersections, int_coord, union = inter.get_info()
plot_intersections(fine_mesh, coarse_mesh, union)

q = Quadrature(3)

phi = Function(2)

coup_op = CouplingOperator(inter, fine_mesh, coarse_mesh)

fine_l2g, coarse_l2g = coup_op.op_l2g_1d()

B = coup_op.compute_b_1d(q, phi)
# print(B)
# print(np.dot(B, np.ones(shape=(3, 1))))

m = MassMatrix(fine_mesh)
M = m.compute_mass_1d(phi, q)
# print(M)
inv_M = np.linalg.inv(M)
# print(inv_M)

L = np.dot(inv_M, B)

# TODO: do a function which creates meshes (parameter, regular o r not) and compute L2-proj
