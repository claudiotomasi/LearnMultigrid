# TODO: make this a class!

from learn_multigrid.mesh.Mesh1D import *


# Map reference -> physical
# x_ref: point on reference
# x_a: left point of physical element
# x_b: right point of physical element
def g_function(x_ref, x_a, x_b):
    # TODO: perche
    return x_a + x_ref * (x_b - x_a)


# Map physical -> reference
# x: point on physical element
# x_a: left point of physical element
# x_b: right point of physical element
def inv_g_function(x, x_a, x_b):
    return (x - x_a)/(x_b - x_a)
