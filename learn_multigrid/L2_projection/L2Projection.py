from learn_multigrid.L2_projection.Intersection import Intersection
from learn_multigrid.L2_projection.CouplingOperator import CouplingOperator
from learn_multigrid.assembly.MassMatrix import MassMatrix
from learn_multigrid.assembly.Quadrature import Quadrature
from learn_multigrid.assembly.ShapeFunction import Function
import numpy as np
import time


class L2Projection:

    def __init__(self, type, fine_mesh, coarse_mesh):
        self.type = type
        self.fine_mesh = fine_mesh
        self.coarse_mesh = coarse_mesh

    def compute_transfer_2d(self):
        fine_mesh = self.fine_mesh
        coarse_mesh = self.coarse_mesh
        inter = Intersection(fine_mesh, coarse_mesh)
        inter.find_intersections2d()
        intersected = inter.get_intersections()

        fine_l2g = np.zeros((intersected.shape[0], 3))

    def compute_transfer_1d(self):
        fine_mesh = self.fine_mesh
        coarse_mesh = self.coarse_mesh
        inter = Intersection(fine_mesh, coarse_mesh)
        start = time.time()
        inter.find_intersections1d()
        intersections, int_coord, _ = inter.get_info()
        end = time.time()
        timeL = end - start
        q = Quadrature(3)

        phi = Function(2)

        coup_op = CouplingOperator(inter, fine_mesh, coarse_mesh)

        B = coup_op.compute_b_1d(q, phi)
        m = MassMatrix(fine_mesh)
        M = m.compute_mass_1d(phi, q)

        method = self.type_to_method(self.type)
        L = method(B, M)
        # BCs: the restricted residual should be 0
        # L[0, :] = 0
        # L[-1, :] = 0

        # # For deleting really small numbers
        # for i in range(L.shape[0]):
        #     for j in range(L.shape[1]):
        #         if L[i][j] < 1e-10:
        #             L[i][j] = 0
        #
        return L, timeL

    def type_to_method(self, type):
        switcher = {
            "L2": self.compute_l2_1d,
            "pseudo": self.compute_pseudo_1d,
            "quasi": self.compute_quasi_1d,
        }
        return switcher.get(type, "Invalid order")

    @staticmethod
    def compute_l2_1d(B, M):
        inv_M = np.linalg.inv(M)
        L = np.dot(inv_M, B)

        return L

    @staticmethod
    def compute_pseudo_1d(B, M):
        # Using lumped mass matrix
        row_sum = np.sum(M, axis=0)
        diag = np.diag(row_sum)
        # inv_M = np.linalg.inv(diag)
        # L = np.dot(inv_M, B)
        L = np.linalg.solve(diag, B)
        return L

    @staticmethod
    def compute_quasi_1d(B, _):
        # Using lumped mass matrix
        row_sums = B.sum(axis=1)
        L = B / row_sums[:, np.newaxis]

        return L
