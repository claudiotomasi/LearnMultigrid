import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

class StiffnessMatrix:

    @staticmethod
    def jacobian(x, y):
        J = np.zeros((2,2))
        J[0, 0] = x[1] - x[0]
        J[0, 1] = x[2] - x[0]
        J[1, 0] = y[1] - y[0]
        J[1, 1] = y[2] - y[0]
        return J

    def __init__(self, mesh):
        self.mesh = mesh
        self.J = self.jacobian
        self.A = lil_matrix([])

    def compute_stiffness_2d(self, d_phi, q):
        n_p = self.mesh.get_np()
        p = self.mesh.get_points()
        conn = self.mesh.get_connections()
        A = lil_matrix((n_p, n_p))
        for l2g in conn:
            x = p[l2g, 0]
            y = p[l2g, 1]

            jac = self.J(x, y)
            d_J = np.linalg.det(jac)
            inv = np.transpose(np.linalg.inv(jac))
            loc_A = self.loc_a_2d(d_J, inv, d_phi, q)
            A[np.ix_(l2g, l2g)] = A[np.ix_(l2g, l2g)] + loc_A
        self.A = A
        return A

    def save(self, path = "../data/matrices/A"):
        x_coo = self.A.tocoo()
        row = x_coo.row
        col = x_coo.col
        data = x_coo.data
        shape = x_coo.shape
        np.savez(path, row=row, col=col, data=data, shape=shape)

    def load(self, path):
        y = np.load(path)
        z = coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
        z = lil_matrix(z)
        self.A = z
        return z

    @staticmethod
    def loc_a_2d(d_J, jac_inv, d_phi, q):
        loc_A = np.zeros((3, 3))
        for i in range(0,3):
            for j in range(0,3):
                loc_A[i, j] = d_J * q.compute_grad(d_phi, jac_inv, np.array([i, j]))
        return loc_A

    def compute_stiffness_1d(self, dphi, q):
        conn = self.mesh.get_connections()
        ne = self.mesh.get_ne()
        n_points = self.mesh.get_np()
        A = np.zeros(shape = (n_points, n_points))

        for i in range(0, ne):
            left = conn[i, 0]
            right = conn[i, 1]
            local_A = self.loc_a_1d(dphi, q, left, right)
            A[np.ix_([i, i+1], [i, i+1])] += local_A
        return A

    @staticmethod
    def loc_a_1d(dphi, q, left, right):
        locA = np.zeros(shape = (2, 2))

        for i in range(0, 2):
            for j in range(0, 2):

                locA[i, j] = 1/(right - left) * q.compute(dphi, np.array([i, j]))
        return locA
