from learn_multigrid.mesh.Mesh1D import *
from learn_multigrid.L2_projection.L2Projection import  *
from learn_multigrid.utilities.plots import *
m = Mesh1DRefinement(coarse_ne = 2, n_ref=3)
m.construct()

m.plot_mesh()


m_c = Mesh1DRefinement(coarse_ne = 3, n_ref = 2)
m_c.construct()
# print(m_c.get_np())
#
# print(m_c.get_connections())
i = Intersection(m, m_c)
i.find_intersections1d()
_, _, u = i.get_info()

tran_op = L2Projection("pseudo", m, m_c)
# #
Q = tran_op.compute_transfer_1d()

plot_intersections(m, m_c, u)

