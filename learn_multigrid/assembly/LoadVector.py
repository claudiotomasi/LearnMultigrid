import numpy as np


class LoadVector:

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
        self.rhs = np.array([])

    def compute_rhs_2d(self, fun, phi, q):
        n_p = self.mesh.get_np()
        p = self.mesh.get_points()
        conn = self.mesh.get_connections()
        rhs = np.zeros((n_p, 1))
        for l2g in conn:
            x = p[l2g, 0]
            y = p[l2g, 1]

            jac = self.J(x, y)
            d_J = np.linalg.det(jac)
            loc_rhs = self.loc_rhs_2d(d_J, fun, phi, q)
            rhs[l2g] = rhs[l2g] + np.reshape(loc_rhs, (3,1))
        self.rhs = rhs
        return rhs

    def save(self, path = "../data/matrices/rhs"):
        rhs = self.rhs
        np.save(path, rhs)

    def load(self, path):
        rhs = np.load(path)
        self.rhs = rhs
        return rhs

    @staticmethod
    def loc_rhs_2d(d_J, fun, phi, q):
        locRHS = np.zeros(shape=(3, 1))

        for i in range(0, 3):
                locRHS[i] = d_J * q.compute_single(phi, i, fun)
        return locRHS

    def compute_rhs_1d(self, f):
        x = self.mesh.get_mesh()
        n_points = self.mesh.get_np()
        b = np.zeros(shape=(n_points, 1))
        for i in range(0, n_points -1):
            local_b = np.array([f(x[i]), f(x[i+1])])*(x[i+1] - x[i])/2
            local_b = np.reshape(local_b, newshape=(2, 1))

            b[np.ix_([i, i + 1])] += local_b
        return b
