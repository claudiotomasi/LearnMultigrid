from learn_multigrid.solvers.Solver import IterativeSolver
from scipy.sparse import diags
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
import numpy as np


class Jacobi(IterativeSolver):

    def __init__(self, matrix, rhs):
        print("Selected Jacobi")
        super().__init__(matrix, rhs)
        self.label = "Jacobi"

    def solve(self, max_iterations = 1000, error = 1e-12, initial_guess = None):
        if initial_guess is None:
            print("You should put an initial guess. Used zero vector")
            self.solution = np.zeros(shape=(self.get_dimension(), 1))
        else:
            self.solution = initial_guess
        A = self.matrix
        d = diags(A.diagonal())
        inv_d = inv(csc_matrix(d))
        # print(inv_d)
        track_res = np.ndarray(shape=(0, 1), dtype = float)
        for i in range(0, max_iterations):
            self.iterations += 1
            self.residual_vector = self.rhs - self.matrix.dot(self.solution)
            self.residual = np.linalg.norm(self.residual_vector)
            # print(self.residual)
            track_res = np.vstack((track_res, self.residual))
            if self.residual <= error:
                print("Reached convergence")
                break
            self.solution += inv_d * self.residual_vector

        self.track_res = track_res

