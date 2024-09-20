from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


class Solver:
    matrix: csc_matrix
    residual: float
    residual_vector: np.array
    dim: int

    def __init__(self, matrix, rhs):
        self.dim = rhs.size
        self.residual_vector = np.empty(shape=rhs.shape)
        self.residual = 0.0
        self.matrix = csc_matrix(matrix)
        self.rhs = rhs
        self.solution = np.empty(shape=rhs.shape)
        self.track_res = np.ndarray(shape=(0, 1), dtype=float)

    def set_matrix(self, matrix):
        self.matrix = matrix

    def get_matrix(self):
        return self.matrix

    def get_residual_vector(self):
        return self.residual_vector

    def get_residual(self):
        return self.residual

    def set_rhs(self, rhs):
        self.rhs = rhs

    def get_rhs(self):
        return self.rhs

    def get_solution(self):
        return self.solution

    def get_dimension(self):
        return self.dim

    def get_track_res(self):
        return self.track_res


class DirectSolver(Solver):

    def __init__(self, matrix, rhs):
        super().__init__(matrix, rhs)

    def solve(self):
        self.solution = np.reshape(spsolve(self.matrix, self.rhs), newshape=(self.dim, 1))
        self.residual_vector = self.rhs - self.matrix.dot(self.solution)
        self.residual = np.linalg.norm(self.residual_vector)


class IterativeSolver(Solver):
    initial_guess: np.array
    residual: float
    step: int
    iterations: int
    error: float

    def __init__(self, matrix, rhs):
        super().__init__(matrix, rhs)
        # self.initial_guess = initial_guess
        self.iterations = 0
        self.label = "Iterative Solver"

    def plot(self, scale = "linear"):
        plt.plot(self.track_res, label=self.label)
        plt.yscale(scale)
        plt.legend()
        plt.title("Residual decreasing")
        plt.ylabel('residual')
        plt.xlabel('iterations')

    def get_iterations(self):
        return self.iterations


