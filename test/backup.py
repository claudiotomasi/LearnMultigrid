from learn_multigrid.solvers.Solver import IterativeSolver
from learn_multigrid.solvers.Jacobi import Jacobi
from learn_multigrid.solvers.GaussSeidel import GaussSeidel
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import inspect
import sys
from cachetools import cached, TTLCache

# TODO: differentiate the computation of interpoaltion op
# TODO: as it is now, the l2proj is computed each time we move from L to L-1


class Multigrid(IterativeSolver):
    # ttl = time in seconds of how much cached stuff will live
    # maxsize = how many objecs
    cache = TTLCache(maxsize=50, ttl=300)

    def __init__(self, matrix, rhs):
        print("Selected Multigrid")
        super().__init__(matrix, rhs)
        self.label = "Multigrid"

    def solve(self, levels=2, smoother="Jacobi", max_iterations=100, error=1e-08, initial_guess=None, cycle="V",
              first_call=False):

        if initial_guess is None:
            print("You should put an initial guess. Used zero vector")
            self.solution = np.zeros(shape=(self.get_dimension(), 1))
        else:
            self.solution = initial_guess
        A = self.matrix
        track_res = np.ndarray(shape=(0, 1), dtype=float)
        # smooth_class = self.smoother_to_method(smoother)
        # if not inspect.isclass(smooth_class):
        #     print("Smoother unknown")
        #     sys.exit(0)
        # s = smooth_class(A, self.rhs)
        track_res = np.ndarray(shape=(0, 1), dtype=float)
        cycle_method = self.cycle_to_method(cycle)
        if callable(cycle_method):
            print("Selected: ", cycle_method)
        else:
            print("Cycle type unknown, exit...")
            sys.exit(0)

        for i in range(0, max_iterations):
            self.iterations += 1
            self.residual_vector = self.rhs - self.matrix.dot(self.solution)
            self.residual = np.linalg.norm(self.residual_vector)
            track_res = np.vstack((track_res, self.residual))
            if self.residual <= error:
                print("Reached convergence")
                break
            # self.solution = self.v_cycle(A, self.solution, self.rhs, smoother, levels)
            self.solution = cycle_method(A, self.solution, self.rhs, smoother, levels, first_call)

        self.track_res = track_res

    def v_cycle(self, A, u0, rhs, smoother, levels, first_call = False):
        levels -= 1
        smooth_class = self.smoother_to_method(smoother)
        s = smooth_class(A, rhs)
        # Pre - Smoothing
        # print("pre-smoothing")
        # print(A, u0)
        s.solve(max_iterations=4, initial_guess=u0)
        u = s.get_solution()
        res = rhs - A.dot(u)
        i = self.interpolator(A.shape[0], first_call)
        res_coarse = np.dot(i.T, res)
        A = A.toarray()
        # A_coarse = np.dot(i.T, A)
        A_coarse = np.linalg.multi_dot([i.T, A, i])
        A_coarse = csr_matrix(A_coarse)
        # A_coarse = np.dot(A_coarse, i)
        # print(A_coarse)

        if levels != 1:
            u_coarse = self.v_cycle(A_coarse, np.zeros(shape=(A_coarse.shape[0], 1)), res_coarse, smoother, levels)
        else:
            # print("Direct Solving")
            u_coarse = np.reshape(spsolve(A_coarse, res_coarse), newshape=(A_coarse.shape[0],1))
            # print(u_coarse)

        # Correction
        u = u + np.dot(i, u_coarse)

        # Post - Smoothing
        # print("post-smoothing")
        s.solve(max_iterations=4, initial_guess=u0)
        u = s.get_solution()

        return u

    @cached(cache)
    def interpolator(self, dimension, _):
        # print("inside: ", dimension)
        rows = dimension
        cols = int(np.floor(dimension/2))
        mat = np.zeros(shape=(rows, cols))
        i = 0
        for j in range(0, cols):
            mat[i, j] = 1
            i += 1
            mat[i, j] = 2
            i += 1
            mat[i, j] = 1
        return mat/2

    def smoother_to_method(self, smoother):
        switcher = {
            "GaussSeidel": GaussSeidel,
            "Jacobi": Jacobi,
        }
        return switcher.get(smoother, "Invalid smoother")

    def cycle_to_method(self, cycle):
        switcher = {
            "V": self.v_cycle,
        }
        return switcher.get(cycle, "Invalid smoother")


class GeometricMG(Multigrid):

    def __init__(self, matrix, rhs):
        print("Selected Geometric Multigrid")
        super().__init__(matrix, rhs)
        self.label = "GeometricMG"

    def interpolator(self, dimension, _):
        print("Geometric interpolator")
        return super().interpolator(dimension, _)


class SemiGeometricMG(Multigrid):

    def __init__(self, matrix, rhs):
        print("Selected Semi - Geometric Multigrid")
        super().__init__(matrix, rhs)
        self.label = "SemiGeometricMG"

    def solve(self, levels=2, smoother="Jacobi", max_iterations=100, error=1e-08, initial_guess=None, cycle="V",
              first_call=True):
        super().solve(levels, smoother, max_iterations, error, initial_guess, cycle, first_call)

    def interpolator(self, dimension, first_call):
        # Here, if first_call True, then return l2 proj.
        # Otherwise, just call super() method since we are in levels below
        print("First_call_Semi: ", first_call)
        return super().interpolator(dimension, first_call)
