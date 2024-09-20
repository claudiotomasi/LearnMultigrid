import numpy as np
from learn_multigrid.assembly.MapReferenceElement import *
from learn_multigrid.assembly.Quadrature import *
from learn_multigrid.assembly.ShapeFunction import *


class CouplingOperator:

    def __init__(self, intersections, fine_mesh, coarse_mesh):
        # intersections.find_intersections1d()
        # before passing intersections, it has already found the intersections
        self.inter = intersections
        self.coarse_mesh = coarse_mesh
        self.fine_mesh = fine_mesh

    def op_l2g_1d(self):
        intersections, _, _ = self.inter.get_info()

        fine_l2g = np.zeros((len(intersections), 2), dtype = int)
        coarse_l2g = np.zeros((len(intersections), 2), dtype=int)

        for i in range(0, len(intersections)):
            el_fine = intersections[i, 0]
            el_coarse = intersections[i, 1]

            fine_l2g[i, :] = [el_fine, el_fine + 1];
            coarse_l2g[i, :] = [el_coarse, el_coarse + 1];

        return fine_l2g, coarse_l2g

    def compute_b_1d(self, q, phi):
        intersections, int_coord, _ = self.inter.get_info()
        conn = self.fine_mesh.get_connections()
        coarse_conn = self.coarse_mesh.get_connections()
        # q = Quadrature(3)
        p = q.get_points()
        # phi = Function(2)
        fine_l2g, coarse_l2g = self.op_l2g_1d()
        B = np.zeros(shape=(self.fine_mesh.get_np(), self.coarse_mesh.get_np()))
        for k in range(0, len(intersections)):
            loc_B = np.zeros(shape=(2, 2))

            intersection = intersections[k, :]
            inters_coord = int_coord[k, :]
            x_a = inters_coord[0]
            x_b = inters_coord[1]

            coord = conn[intersection[0], :]
            fine_x_a = coord[0]
            fine_x_b = coord[1]

            coarse_coord = coarse_conn[intersection[1], :]
            coarse_x_a = coarse_coord[0]
            coarse_x_b = coarse_coord[1]

            physic_p = g_function(p, x_a, x_b)

            fine_ref = inv_g_function(physic_p, fine_x_a, fine_x_b)
            coarse_ref = inv_g_function(physic_p, coarse_x_a, coarse_x_b)

            f_nodes = fine_l2g[k, :]
            c_nodes = coarse_l2g[k, :]
            for i in range(0, 2):
                for j in range(0, 2):
                    # phi_i = phi.get_functions()[i]
                    # phi_j = phi.get_functions()[j]
                    loc_B[i, j] = (x_b - x_a) * q.compute_inter(phi, [i, j], fine_ref, coarse_ref)
            B[np.ix_(f_nodes, c_nodes)] += loc_B
        return B
