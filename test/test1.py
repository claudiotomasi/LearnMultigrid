import sys
import os


def f1(x):
    return 1-x


sys.path.append('../')
# from learn_multigrid import mesh
from learn_multigrid.mesh.Mesh1D import *
from learn_multigrid.assembly.MapReferenceElement import *
from learn_multigrid.assembly.Quadrature import *
from learn_multigrid.assembly.ShapeFunction import *
from learn_multigrid.L2_projection.Intersection import *
from learn_multigrid.L2_projection.CouplingOperator import *
from learn_multigrid.utilities.plots import *


fine_mesh = Mesh1D(regular = True, ne = 4)
fine_mesh.construct()
fine_x = fine_mesh.get_mesh()
# fine_mesh.plot_mesh()
# print(x)

coarse_mesh = Mesh1D(regular = True, ne = 2)
coarse_mesh.construct()
coarse_x = coarse_mesh.get_mesh()
# coarse_mesh.plot_mesh()
# print(coarse_x)

# print(np.union1d(fine_x, coarse_x))
# print(inv_g_function(0.75, 0.5, 1))
# print(coarse_mesh.get_connections())

# print(fine_mesh.get_connections())

inter = Intersection(fine_mesh, coarse_mesh)
inter.find_intersections1d()
intersections, int_coord, union = inter.get_info()
plot_intersections(fine_mesh, coarse_mesh, union)

# print(intersections, "\n", int_coord,  "\n", union)
conn = fine_mesh.get_connections()
intersection = intersections[0, :]
coord = conn[intersection[0], :]
x_a = coord[0]
x_b = coord[1]
physic_p = g_function(0.5, x_a, x_b)
# print(physic_p)

p = np.array([0.11270166537926, 0.50000000000000, 0.88729833462074])
physic_p = g_function(p, x_a, x_b)

# print(physic_p)
fine_ref = inv_g_function(physic_p, x_a, x_b)
# print(fine_ref)
coarse_conn = coarse_mesh.get_connections()
coarse_coord = coarse_conn[intersection[1], :]
coarse_x_a = coarse_coord[0]
coarse_x_b = coarse_coord[1]
coarse_ref = inv_g_function(physic_p, coarse_x_a, coarse_x_b)
# print(coarse_ref)

coup_op = CouplingOperator(inter, fine_mesh, coarse_mesh)
fine_l2g, coarse_l2g = coup_op.op_l2g_1d()

q = Quadrature(3)
# print(q.get_points())


phi = Function(2)

# res = q.compute_inter(phi, [0, 0], np.array([0.1127, 0.5000, 0.8873]), np.array([0.5564, 0.7500, 0.9436]))
# print(res)



# print(phi.evaluate(np.array([2, -2, 4, -4])))


B = coup_op.compute_b(q, phi)
# print(B)
print(np.dot(B, np.ones(shape=(3, 1))))

