from learn_multigrid.solvers.Solver import IterativeSolver
import numpy as np


class CG(IterativeSolver):

    def __init__(self, matrix, rhs):
        print("Selected CG")
        super().__init__(matrix, rhs)
        self.label = "CG"

    def solve(self, max_iterations = 1000, error = 1e-08, initial_guess = None):
        if initial_guess is None:
            print("You should put an initial guess. Used zero vector")
            self.solution = np.zeros(shape=(self.get_dimension(), 1))
        else:
            self.solution = initial_guess

        A = self.matrix

        track_res = np.ndarray(shape=(0, 1), dtype=float)
        self.residual_vector = self.rhs - self.matrix.dot(self.solution)
        self.residual = np.linalg.norm(self.residual_vector)
        track_res = np.vstack((track_res, self.residual))
        r = self.residual_vector
        p = self.residual_vector
        x = self.solution
        for i in range(0, max_iterations):
            self.iterations += 1
            r2 = np.asscalar(r.T @ r)
            Ap = A.dot(p)
            pAp = np.asscalar(p.T @ Ap)

            alpha = r2/pAp
            x = x + alpha * p

            r = r - alpha * A @ p

            self.solution = x
            self.residual_vector = r
            self.residual = np.linalg.norm(r)
            track_res = np.vstack((track_res, self.residual))
            if self.residual <= error:
                print("Reached convergence")
                break

            beta = np.asscalar(r.T @ r) / r2
            p = r + beta * p

        self.track_res = track_res
