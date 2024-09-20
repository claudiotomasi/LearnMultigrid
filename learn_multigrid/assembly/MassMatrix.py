import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

class MassMatrix:

    @staticmethod
    def jacobian(x, y):
        J = np.zeros((2, 2))
        J[0, 0] = x[1] - x[0]
        J[0, 1] = x[2] - x[0]
        J[1, 0] = y[1] - y[0]
        J[1, 1] = y[2] - y[0]
        return J

    def __init__(self, mesh):
        self.mesh = mesh
        self.J = self.jacobian
        self.M = lil_matrix([])

    def compute_mass_2d(self, phi, q):
        n_p = self.mesh.get_np()
        p = self.mesh.get_points()
        conn = self.mesh.get_connections()
        M = lil_matrix((n_p, n_p))
        for l2g in conn:
            x = p[l2g, 0]
            y = p[l2g, 1]

            jac = self.J(x, y)
            d_J = np.linalg.det(jac)
            loc_M = self.loc_m_2d(d_J, phi, q)
            M[np.ix_(l2g, l2g)] = M[np.ix_(l2g, l2g)] + loc_M
        self.M = M
        return M

    def save(self, path = "../data/matrices/M"):
        x_coo = self.M.tocoo()
        row = x_coo.row
        col = x_coo.col
        data = x_coo.data
        shape = x_coo.shape
        np.savez(path, row=row, col=col, data=data, shape=shape)

    def load(self, path):
        y = np.load(path)
        z = coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
        z = lil_matrix(z)
        self.M = z
        return z

    @staticmethod
    def loc_m_2d(d_J, phi, q):
        locM = np.zeros(shape=(3, 3))

        for i in range(0, 3):
            for j in range(0, 3):
                locM[i, j] = d_J * q.compute(phi, np.array([i, j]))
        return locM

    def compute_mass_1d(self, phi, q):
        conn = self.mesh.get_connections()
        ne = self.mesh.get_ne()
        n_points = self.mesh.get_np()
        M = np.zeros(shape = (n_points, n_points))

        for i in range(0, ne):
            left = conn[i, 0]
            right = conn[i, 1]
            local_M = self.loc_m_1d(phi, q, left, right)
            M[np.ix_([i, i+1], [i, i+1])] += local_M
        return M

    @staticmethod
    def loc_m_1d(phi, q, left, right):
        locM = np.zeros(shape = (2, 2))

        for i in range(0, 2):
            for j in range(0, 2):

                locM[i, j] = (right - left) * q.compute(phi, np.array([i, j]))
        return locM
