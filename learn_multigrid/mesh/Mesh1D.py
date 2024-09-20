# This class represents a 1D Mesh
# TODO: it should be a composite of Element1D
import numpy as np
import matplotlib.pyplot as plt


class Mesh1D:
    ne: int
    np: int
    h: float

    def __init__(self, regular=True, ne=0):
        print("Initializing Mesh 1D...")
        self.regular = regular
        self.ne = ne
        self.np = ne + 1
        self.h = 1 / ne
        self.x = np.array([])
        self.conn = np.ndarray(shape=(self.ne, 2))

    def construct(self):
        if self.is_regular():
            self.construct_regular()
        else:
            self.construct_irregular()
        self.connection_matrix()

    def construct_regular(self):
        self.x = np.linspace(0, 1, self.np)

    def construct_irregular(self):
        h = self.h
        tmp = np.linspace(0, 1, self.np)
        # a = 0
        # b = 0.6 * h
        b = h / 4
        a = h / 8
        for i in range(1, self.np - 1):
            r = (b - a) * np.random.rand() + a
            # tmp[i] = tmp[i] - r
            tmp[i] = tmp[i] - r
        self.x = tmp

    def is_regular(self):
        return self.regular

    def connection_matrix(self):
        conn = self.conn
        x = self.x
        for i in range(0, len(conn)):
            conn[i, :] = [x[i], x[i + 1]]
        self.conn = conn

    def get_connections(self):
        return self.conn

    def get_ne(self):
        return self.ne

    def get_np(self):
        return self.np

    def get_mesh(self):
        return self.x

    def plot_mesh(self):
        x = self.x
        y = np.zeros(len(x))
        plt.plot(x, y, 'ro')
        plt.grid()
        plt.title('Mesh')
        plt.xticks(np.arange(0, 1.1, step=0.1))  # To show more points on grid
        plt.show()


class Mesh1DRefinement(Mesh1D):

    def __init__(self, coarse_ne = 2, n_ref = 0):
        self.ne = coarse_ne
        self.np = coarse_ne + 1
        self.h = 1 / coarse_ne
        self.x = np.linspace(0, 1, self.np)
        self.conn = np.ndarray(shape=(self.ne, 2))
        self.n_ref = n_ref

    def construct(self):
        x = self.x
        ne = self.ne
        for i in range(0, self.n_ref):
            ne = ne * 2
        self.ne = ne
        self.np = ne + 1
        self.h = 1 / ne
        self.x = np.linspace(0, 1, self.np)
        self.conn = np.ndarray(shape=(self.ne, 2))
        super().connection_matrix()
