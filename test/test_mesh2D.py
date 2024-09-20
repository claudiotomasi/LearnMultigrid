from learn_multigrid.mesh.Mesh2D import *
from learn_multigrid.assembly.StiffnessMatrix import *
from learn_multigrid.assembly.LoadVector import *
from learn_multigrid.assembly.LoadFunction import *
from learn_multigrid.assembly.MassMatrix import *
from learn_multigrid.assembly.Quadrature import *
from learn_multigrid.assembly.ShapeFunction import *


def f(x):
    # return np.sin(2 * np.pi * x)
    return -1


mesh = Mesh2D(1)
# mesh.plot_mesh()

mesh.refine(regular = True)

mesh.plot_mesh()
stiff = StiffnessMatrix(mesh)
mass = MassMatrix(mesh)
load = LoadVector(mesh)
fun = LoadFunction(f)

q = Quadrature2D(3)

d_phi = GradientTriangle(1)
phi = FunctionTriangle(1)

A = stiff.compute_stiffness_2d(d_phi, q)
M = mass.compute_mass_2d(phi, q)
rhs = load.compute_rhs_2d(fun, phi, q)

# Find Boundary nodes
p = mesh.p
h_border = np.logical_or(p[:, 0] == 0, p[:, 0] == 1)
v_border = np.logical_or(p[:, 1] == 0, p[:, 1] == 1)
border = np.logical_or(h_border, v_border)
nodes = np.where(border)[0]
i = np.eye(len(p))
A[nodes, :] = i[nodes, :]
rhs[nodes] = 0

# new_p, new_conn = mesh.embedding()
embed = mesh.embedding()
embed.plot_mesh()
embed_mass = MassMatrix(embed)
M_2 = embed_mass.compute_mass_2d(phi,q)
