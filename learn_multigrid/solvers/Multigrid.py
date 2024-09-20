from learn_multigrid.solvers.Solver import IterativeSolver
from learn_multigrid.solvers.Jacobi import Jacobi
from learn_multigrid.solvers.CG import CG
from learn_multigrid.solvers.GaussSeidel import GaussSeidel
import numpy as np
from scipy.sparse.linalg import spsolve
from pyamg.relaxation.relaxation import gauss_seidel
from scipy.sparse import lil_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
from scipy import linalg
from scipy.sparse.linalg import dsolve
from scipy.sparse import csr_matrix
import inspect
import sys
import copy
from cachetools import cached, TTLCache
import time

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

    def solve(self, levels=2, smoother="Jacobi", smooth_steps=1, max_iterations=100, error=1e-08, initial_guess=None, cycle="V",
              first_call=False):

        if initial_guess is None:
            print("You should put an initial guess. Used zero vector")
            self.solution = np.zeros(shape=(self.get_dimension(), 1))
        else:
            self.solution = initial_guess
        A = self.matrix
        # track_res = np.ndarray(shape=(0, 1), dtype=float)
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
            print("It: ", self.iterations)
            self.residual_vector = self.rhs - self.matrix.dot(self.solution)
            self.residual = np.linalg.norm(self.residual_vector)
            if self.iterations <= 1:
                self.residual_vector = np.ones(shape=self.solution.shape)
                self.residual = np.linalg.norm(self.residual_vector)
            track_res = np.vstack((track_res, self.residual))
            print(self.residual)
            if self.residual <= error:
                # print("Reached convergence level: ", levels)
                break
            # self.solution = self.v_cycle(A, self.solution, self.rhs, smoother, levels)
            self.solution = cycle_method(A, self.solution, self.rhs, smoother, smooth_steps, error, levels, first_call)

        self.track_res = track_res

    def v_cycle(self, A, u0, rhs, smoother, smooth_steps, error, levels, first_call = False):
        levels -= 1
        smooth_class = self.smoother_to_method(smoother)
        s = smooth_class(A, rhs)
        # Pre - Smoothing
        # print("pre-smoothing")
        # print(A, u0)

        # s.solve(max_iterations=smooth_steps, initial_guess=u0, error = sys.float_info.min)
        # u = s.get_solution()

        gauss_seidel(A, u0, rhs, iterations=smooth_steps)
        u = copy.copy(u0)
        res = rhs - A.dot(u)
        i = self.interpolator(A.shape[0], first_call)
        # res_coarse = np.dot(i.T, res)
        res_coarse = i.T @ res
        # A = A.toarray()
        # A_coarse = np.dot(i.T, A)
        # A_coarse = np.linalg.multi_dot([i.T, A, i])
        A_coarse = i.T @ A @ i
        A_coarse = csr_matrix(A_coarse)
        # A_coarse = np.dot(A_coarse, i)
        # print(A_coarse)

        if levels != 1:
            u_coarse = self.v_cycle(A_coarse, np.zeros(shape=(A_coarse.shape[0], 1)), res_coarse, smoother,
                                    smooth_steps, error, levels)
        else:
            u_coarse = np.reshape(spsolve(A_coarse, res_coarse, use_umfpack = False), newshape=(A_coarse.shape[0],1))
            # print("End Direct")
            # u_coarse = np.reshape(linalg.solve(A_coarse.todense(), res_coarse, sym_pos=True), newshape=(
            # A_coarse.shape[0],1))

            # print(u_coarse)

        # Correction
        # u = u + np.dot(i, u_coarse)
        u = u + i @ u_coarse

        # Post - Smoothing
        # print("post-smoothing")
        # s.solve(max_iterations=smooth_steps, initial_guess=u, error = sys.float_info.min)
        # u = s.get_solution()
        gauss_seidel(A, u, rhs, iterations=smooth_steps)
        # u = copy.copy(u0)

        return u

    @cached(cache)
    def interpolator(self, dimension, _):
        print("inside: ", dimension)
        rows = dimension
        cols = int(np.floor((dimension-1)/2)) + 1
        mat = np.zeros(shape=(rows, cols))
        i = 1
        for j in range(1, cols-1):
            mat[i, j] = 1
            i += 1
            mat[i, j] = 2
            i += 1
            mat[i, j] = 1
        mat[0, 0] = 2
        mat[1, 0] = 1
        mat[-1, -1] = 2
        mat[-2, -1] = 1
        # mat[:, 0] = 0
        # mat[0, 0] = 1
        # mat[:, -1] = 0
        # mat[-1, -1] = 1
        return mat/2

    def smoother_to_method(self, smoother):
        switcher = {
            "GaussSeidel": GaussSeidel,
            "Jacobi": Jacobi,
            "CG": CG,
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

    def __init__(self, matrix, rhs, l2_proj):
        print("Selected Semi - Geometric Multigrid")
        super().__init__(matrix, rhs)
        self.label = "SemiGeometricMG"
        self.l2_proj = csr_matrix(l2_proj)

    def solve(self, levels=2, smoother="Jacobi", smooth_steps = 1, max_iterations=100, error=1e-08, initial_guess=None, cycle="V",
              first_call=True):
        super().solve(levels, smoother, smooth_steps, max_iterations, error, initial_guess, cycle, first_call)

    def interpolator(self, dimension, first_call):
        # Here, if first_call True, then return l2 proj.
        # Otherwise, just call super() method since we are in levels below
        # print("First_call_Semi: ", first_call)
        if first_call:
            print("Semi")
            return self.l2_proj
        else:
            print("Geom")
            return super().interpolator(dimension, first_call)


class NeuralMG(Multigrid):

    def __init__(self, matrix, rhs, model, M, std, mean):
        print("Selected NN Multigrid")
        super().__init__(matrix, rhs)
        self.label = "NeuralMG"
        self.model = model
        self.M = M
        self.std = std
        self.mean = mean

    def solve(self, levels=2, smoother="Jacobi", smooth_steps=1, max_iterations=100, error=1e-08, initial_guess=None, cycle="V",
              first_call=False):

        if initial_guess is None:
            print("You should put an initial guess. Used zero vector")
            self.solution = np.zeros(shape=(self.get_dimension(), 1))
        else:
            self.solution = initial_guess
        A = self.matrix
        track_res = np.ndarray(shape=(0, 1), dtype=float)
        cycle_method = self.cycle_to_method(cycle)
        if callable(cycle_method):
            print("Selected: ", cycle_method)
        else:
            print("Cycle type unknown, exit...")
            sys.exit(0)

        for i in range(0, max_iterations):
            self.iterations += 1
            print("It: ", self.iterations)
            self.residual_vector = self.rhs - self.matrix.dot(self.solution)
            self.residual = np.linalg.norm(self.residual_vector)
            if self.iterations <= 1:
                self.residual_vector = np.ones(shape=self.solution.shape)
                self.residual = np.linalg.norm(self.residual_vector)
            track_res = np.vstack((track_res, self.residual))
            print(self.residual)
            if self.residual <= error:
                # print("Reached convergence level: ", levels)
                break
            # self.solution = self.v_cycle(A, self.solution, self.rhs, smoother, levels)
            self.solution = cycle_method(A, self.M, self.solution, self.rhs, smoother, smooth_steps, error, levels, first_call)

        self.track_res = track_res

    def v_cycle(self, A, M, u0, rhs, smoother, smooth_steps, error, levels, first_call = False):
        levels -= 1
        smooth_class = self.smoother_to_method(smoother)
        s = smooth_class(A, rhs)
        # Pre - Smoothing
        # print("pre-smoothing")
        # print(A, u0)

        # s.solve(max_iterations=smooth_steps, initial_guess=u0, error = sys.float_info.min)
        # u = s.get_solution()

        gauss_seidel(A, u0, rhs, iterations=smooth_steps)
        u = copy.copy(u0)

        res = rhs - A.dot(u)
        i = self.transfer_op(M)
        # res_coarse = np.dot(i.T, res)
        res_coarse = i.T @ res
        # A = A.toarray()
        # A_coarse = np.dot(i.T, A)
        # A_coarse = np.linalg.multi_dot([i.T, A, i])
        # TODO: It is possibile that with more levels, A and M coarse cannot be csr since we need them for interpolator
        # TODO: possibile solution -> prepare first the interpolators saving them in cache, then move to solve  phase
        A_coarse = i.T @ A @ i
        A_coarse = csr_matrix(A_coarse)

        # M_coarse = np.linalg.multi_dot([i.T, M, i])
        M_coarse = i.T @ M @ i
        M_coarse = csr_matrix(M_coarse)
        M_coarse = M_coarse.toarray()
        # A_coarse = np.dot(A_coarse, i)
        # print(A_coarse)

        if levels != 1:
            print("Going on level: ", levels)
            u_coarse = self.v_cycle(A_coarse, M_coarse, np.zeros(shape=(A_coarse.shape[0], 1)), res_coarse, smoother,
                                    smooth_steps, error, levels)
        else:
            print("Direct Solving")
            u_coarse = np.reshape(spsolve(A_coarse, res_coarse, use_umfpack = False), newshape=(A_coarse.shape[0],1))
            # u_coarse = np.reshape(linalg.solve(A_coarse.todense(), res_coarse, sym_pos=True), newshape=(
            # A_coarse.shape[0],1))

            # print(u_coarse)

        # Correction
        # u = u + np.dot(i, u_coarse)
        u = u + i @ u_coarse

        # Post - Smoothing
        # print("post-smoothing")
        # s.solve(max_iterations=smooth_steps, initial_guess=u, error = sys.float_info.min)
        # u = s.get_solution()
        gauss_seidel(A, u, rhs, iterations=smooth_steps)

        return u

    cache_transfer = TTLCache(maxsize=50, ttl=300)

    # @cached(cache_transfer)
    def transfer_op(self, M):
        print("Computing Neural Transfer op.")
        data_M = self.prepare_nn_input(M)
        data_M = (data_M - self.mean) / self.std
        Q = self.construct_B(data_M, M)
        return Q

    def prepare_nn_input(self, mass):
        # prepare NN input
        M = copy.copy(mass)
        dim = M.shape[0]
        dim_ne = dim - 1
        k = 0
        data_M = np.zeros(shape=(int(dim_ne / 2) - 1, 7))

        for i in (range(1, M.shape[0] - 2, 2)):
            locM_2 = M[i:i + 3, i - 1:i + 4]
            locM = copy.copy(locM_2)
            locM[0, 3] = locM[0, 4] = 0
            locM[1, 0] = locM[1, 1] = 0
            locM[1, 4] = locM[2, 0] = 0
            locM[2, 1] = locM[2, 2] = 0
            row = locM[np.nonzero(locM)]
            # _, idx = np.unique(locM, return_index=True)
            # row = locM[np.sort(idx)]
            data_M[k, :] = row
            k = k + 1

        return data_M

    def construct_B(self, data_M, M):
        # M = self.M
        dim = M.shape[0]
        model = self.model
        B = np.zeros(shape=(dim, int((dim - 1) / 2) + 1))
        res = model.predict(data_M)

        i = 0
        j = 0
        for k in range(0, res.shape[0]):
            patch = res[k]
            # B[i+2, j] = np.mean([patch[2], B[i+2, j]])
            B[i + 2, j] = patch[2]

            middle = patch[4:7]
            # middle[0] = np.mean([B[i, j + 1], middle[0]])
            B[i + 1:i + 4, j + 1] = middle

            B[i + 2, j + 2] = patch[8]

            i = i + 2
            j = j + 1

        sum_row_M = M.sum(axis=1)
        sum_row_B = B.sum(axis=1)
        diff = sum_row_M - sum_row_B
        B[0:2, 0] = diff[0:2]
        B[-2:, -1] = diff[-2:]


        # Now to Q
        row_sums = B.sum(axis=1)
        Q = B / row_sums[:, np.newaxis]

        return Q


class NeuralMG_2D(Multigrid):

    def __init__(self, matrix, rhs, model, M, std, mean):
        print("Selected 2D Neural Multigrid")
        super().__init__(matrix, rhs)
        self.label = "NeuralMG"
        self.model = model
        self.M = M
        self.std = std
        self.mean = mean
        self.l_hierarchy = []

    @staticmethod
    def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

    @staticmethod
    def get_conn(mat):
        matrix = mat[:]
        matrix.setdiag(0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    matrix[i, j] = 1
        return matrix.toarray()

    @staticmethod
    def coarsening(_conn):
        conn = _conn[:]
        conn.setdiag(0)
        conn = conn.toarray()
        CC = []
        FF = []
        R = list(range(0, conn.shape[0]))
        CC_neighs = {}
        FF_neighs = {}
        while R:
            index = min(R)
            R = list(set(R) - {index})
            CC.append(index)
            row = conn[index, :]
            row = np.where(row > 0)[0]
            CC_neighs[index] = row
            neigh = list(set(row) & set(R))
            R = list(set(R) - set(neigh))

            rows = conn[neigh, :]
            for j in range(0, len(rows)):
                neigh_row = rows[j]
                FF_neighs[neigh[j]] = np.where(neigh_row > 0)[0]
            FF.extend(neigh)

        return CC, FF, CC_neighs, FF_neighs

    @staticmethod
    def scaling_vnodes(_neighs):
        switcher = {
            2: [6, 1],
            3: [3, 2],
            4: [2, 3],
            5: [1.5, 4]
        }
        return switcher.get(_neighs, "Invalid month")

    def create_virtual_nodes(self, row, node_M):
        # row = row.toarray()[0]
        actual_neigh = len(row)
        virtual_neighs = []
        if actual_neigh < 6:
            # create virtual neighbours
            scale = self.scaling_vnodes(actual_neigh)
            up = scale[0]
            down = scale[1]
            row = np.append(row, np.ones((1, 6 - actual_neigh)) * node_M / down)
            # Now, create virtual 'rows' of those neighbours to append in patch
            n_virtual = 6 - actual_neigh
            diag_el = node_M * up
            virtual_neigh_row = np.ones((1, 5)) * node_M / down
            sample_virt_neigh = np.append(diag_el, virtual_neigh_row)
            virtual_neighs = np.tile(sample_virt_neigh, (1, n_virtual))[0]
        return virtual_neighs, row

    def direct_neighs(self, where, mat, c_node):
        B = []
        neighs = np.ones(36) * -1
        jj = 0
        for neighbour in where:
            tt = time.time()
            row_neigh = copy.copy(mat[neighbour, :])
            row_neigh[0, c_node] = 0
            B.extend(row_neigh.nonzero()[1])
            # print("TTT: ", time.time() - tt)
            neigh_node = row_neigh[0, neighbour]
            row_neigh[0, neighbour] = 0
            row_neigh = row_neigh[row_neigh > 0]

            # Fill missing entries in existing nodes
            actual_neigh = row_neigh.getnnz()
            row_neigh = row_neigh.toarray()[0]
            if actual_neigh > 5:
                while actual_neigh > 5:
                    pos = np.where(row_neigh == np.min(row_neigh))[0][0]
                    row_neigh = np.delete(row_neigh, [pos])
                    actual_neigh = len(row_neigh)
            # 5 because we erase row[c_node]
            if actual_neigh < 5:
                # create virtual neighbours
                # +1 because we delete the c_node value, but we still need to consider it
                scale = self.scaling_vnodes(actual_neigh + 1)
                up = scale[0]
                down = scale[1]
                row_neigh = np.append(row_neigh, np.ones((1, 5 - actual_neigh)) * neigh_node / down)
            neighs[jj] = neigh_node
            neighs[jj + 1: jj + len(row_neigh) + 1] = row_neigh
            jj = jj + 6

        return B, neighs

    def intersecting_rows(self, where, mat, c_node, CC, fill_idx):
        pos = 13
        for el in where:
            row_el = copy.copy(mat[el, :])
            row_el[0, c_node] = 0
            row_el[0, el] = 0
            row_neigh = row_el.nonzero()[1]
            row_el = row_el[row_el > 0]
            actual_neigh = len(row_neigh)
            if actual_neigh > 5:
                dense_row_el = row_el.toarray()[0]
                while actual_neigh > 5:
                    pos_to_del = np.where(dense_row_el == min(dense_row_el))[0][0]
                    dense_row_el = np.delete(dense_row_el, [pos_to_del])
                    row_neigh = np.delete(row_neigh, [pos_to_del])
                    actual_neigh = len(dense_row_el)

            where_with_coarse = []
            for kk in row_neigh:
                neigh_row = copy.copy(mat[kk, :])
                wh = neigh_row.nonzero()[1]
                neigh_row = neigh_row[neigh_row > 0]
                actual_neigh = neigh_row.getnnz()
                # Since we do not erase nodes we need to consider all 7 entries
                if actual_neigh > 7:
                    dense_neigh_row = neigh_row.toarray()[0]
                    while actual_neigh > 7:
                        pos_to_del = np.where(dense_neigh_row == min(dense_neigh_row))[0][0]
                        dense_neigh_row = np.delete(dense_neigh_row, [pos_to_del])
                        wh = np.delete(wh, [pos_to_del])
                        actual_neigh = len(dense_neigh_row)
                where_with_coarse.extend(wh)

            where_with_coarse = sorted(set(where_with_coarse), key=where_with_coarse.index)
            # st = time.time()
            w_w_c = set(where_with_coarse)
            set_cc = set(CC)
            where_with_coarse = sorted(w_w_c & set_cc, key=where_with_coarse.index)

            # where_with_coarse = [x for x in where_with_coarse if x in CC]
            # print("TIME intersecting rows: ", time.time() - st)
            where_with_coarse = self.diff(where_with_coarse, [c_node])

            while len(where_with_coarse) > 3:
                # pick = random.randint(0, len(where_with_coarse) - 1)
                pick = 0
                where_with_coarse = np.delete(where_with_coarse, [pick])

            fill_idx[pos: pos + len(where_with_coarse)] = where_with_coarse
            pos = pos + 3

        return fill_idx

    def single_extraction(self, row, where, mat, c_node, CC, node_M):
        patch = np.ones(43) * -1
        fill_idx = np.ones(31) * -1

        st = time.time()
        B, neighs = self.direct_neighs(where, mat, c_node)
        # print("direct_neigh: ", time.time() - st)
        # This remove duplicates maintaining original order
        B = sorted(set(B), key=B.index)
        # print(B)
        st_diff = time.time()
        D = self.diff(self.diff(B, where), [c_node])
        # _auxset = set(D)
        # where_nOfN = [x for x in D if x in CC]

        set_D = set(D)
        set_cc = set(CC)
        where_nOfN = sorted(set_D & set_cc, key=D.index)
        # print("time sigle_extr intersection: ", time.time() - st_diff)
        # print(where_nOfN)

        patch[0] = node_M

        virtual_neighs, row = self.create_virtual_nodes(row, node_M)

        if len(virtual_neighs) > 0:
            neighs[-len(virtual_neighs):] = virtual_neighs

        patch[1: len(row) + 1] = row

        patch[7:] = neighs

        fill_idx[0] = c_node
        fill_idx[1:len(where) + 1] = where
        fill_idx[7: 7 + len(where_nOfN)] = where_nOfN

        # INTERSECTING ROWS
        # For each neighbour of c_node, i.e, for each index in 'where', we need 3 intersecting values
        # For c_node, its row was given by taking neighs(neighs(c_node)), minus 'where' minus 'c_node'
        # We need to do the same but for neighbour(c_node), thus for each element in 'where'
        # So, we start from nOfN which represents the neigh(where) and keep asking for its neighbours
        st = time.time()
        fill_idx = self.intersecting_rows(where, mat, c_node, CC, fill_idx)
        # print("intersect coarse: ", time.time() - st)
        return patch, fill_idx

    def extract_patches(self, CC, _mat):
        # For each c_node I need to save the index where to fill B.
        # I need 'where' -> direct neighbours of c_node in fine grid
        # And 'where_nOfN -> direct neighbours of c_node in coarse grid
        #   it contains the neighbors of the neighbors of node
        #   If we intersect with c_index, we get the correct column indexes to fill row_B

        # Then I need 'intersecting' -> rows intersecting the column of B
        _patches = np.zeros((len(CC), 43))
        _fill_idxs = np.zeros((len(CC), 31), dtype=int)
        addition_patches = np.zeros((1, 43))
        addition_idx = np.zeros((1, 31), dtype=int)
        ii = 0
        mat = copy.copy(_mat)

        to_add = 0
        for c_node in CC:
            row = copy.copy(mat[c_node, :])
            row[0, c_node] = 0
            row = row[row > 0]
            # actual_neigh = len(row)
            actual_neigh = row.getnnz()
            additional = (actual_neigh - 6)
            if additional > 0:
                to_add = to_add + additional
        conta_nodi = 0
        for c_node in CC:
            conta_nodi += 1
            # patch consists of:
            # 1st is node_M - in pos 0
            # 2nd is row - in pos [1:7] (meaning until patch[6])
            # 3rd is neighs - pos [7:]
            # neighs consists of six blocks where:
            # 1st is neigh_node
            # 2nd row_neigh
            # patch = np.ones(43) * -1
            # fill_idx = np.ones(31) * -1
            # we need to use copy.copy(), otherwise we modify actual M
            row = copy.copy(mat[c_node, :])
            node_M = row[0, c_node]
            row[0, c_node] = 0
            # where = np.where(row)[0]

            where = row.nonzero()[1]
            # print(where)
            row = row[row > 0]

            actual_neigh = row.getnnz()
            additional = actual_neigh - 6
            row_orig = copy.copy(row)
            where_origin = copy.copy(where)
            # while actual_neigh > 6:
            st = time.time()
            # Works only if neighs are 6 + 1
            if additional > 0:
                dense_row_origin = row_orig.toarray()[0]
                ordered = np.argsort(dense_row_origin)
                add_patches = np.zeros((additional + 1, 43))
                add_fill_idxs = np.zeros((additional + 1, 31), dtype=int)
                for add in range(0, additional + 1):
                    # pos = np.where(row_orig == np.min(row_orig))[0][0]
                    pos = ordered[add:add + additional]
                    row = np.delete(dense_row_origin, [pos])
                    where = np.delete(where_origin, [pos])
                    # actual_neigh = len(row)

                    patch_add, fill_idx_add = self.single_extraction(row, where, mat, c_node, CC, node_M)
                    add_patches[add, :] = patch_add
                    add_fill_idxs[add, :] = fill_idx_add
                _patches[ii, :] = add_patches[0, :]
                _fill_idxs[ii, :] = add_fill_idxs[0, :]
                addition_patches = np.vstack((addition_patches, add_patches[1:, :]))
                addition_idx = np.vstack((addition_idx, add_fill_idxs[1:, :]))

            else:
                st_single = time.time()
                patch, fill_idx = self.single_extraction(row.toarray()[0], where, mat, c_node, CC, node_M)
                # print("time single: ", time.time() - st_single)
                _patches[ii, :] = patch
                _fill_idxs[ii, :] = fill_idx

            print("additional: ", additional, "coarse node: ", conta_nodi, " on total: ", len(CC),
                  "time: ", time.time() - st)
            ii = ii + 1
        _patches = np.vstack((_patches, addition_patches[1:, :]))
        _fill_idxs = np.vstack((_fill_idxs, addition_idx[1:, :]))
        return _patches, _fill_idxs

    @staticmethod
    def map_coarse(_C):
        dim = len(_C)
        enum = list(range(0, dim))
        _mapp = dict(zip(_C, enum))
        return _mapp

    @staticmethod
    def fill_B(_res, _idx_fill, total_size, mapping, CC):
        d_neighs = {}
        _B = np.zeros((total_size, len(CC)))
        # _B = lil_matrix((total_size, len(CC)))
        for j in range(0, len(_res)):
            print("pred ", j, "of ", len(_res))
            pred = _res[j]
            where = _idx_fill[j]

            node = where[0]
            node_coarse = [mapping.get(key) for key in where[0:1]][0]
            col_B = where[1:7]
            # col_B = [mapping.get(key) for key in where[1:7]]

            # col_B indexes are the ones in fine mesh, so just compy them, not map in coarse_indexes
            col_B = col_B[col_B >= 0]
            # row_B = where[7:13]
            row_B = np.array([mapping.get(key) for key in where[7:13]])
            # row_B = row_B[row_B >= 0]
            row_B = row_B[row_B != np.array(None)].astype(int)
            d_neighs[node_coarse] = row_B

            # inters = where[13:]
            if _B[node, node_coarse] == 0:
                _B[node, node_coarse] = pred[0]
            else:
                _B[node, node_coarse] = np.mean((_B[node, node_coarse], pred[0]))

            _B[col_B[_B[col_B, node_coarse] == 0], node_coarse] = np.nan
            _B[col_B, node_coarse] = np.nanmean((_B[col_B, node_coarse], pred[1:1 + len(col_B)]), axis=0)

            _B[node, row_B[_B[node, row_B] == 0]] = np.nan
            _B[node, row_B] = np.nanmean((_B[node, row_B], pred[7:7 + len(row_B)]), axis=0)

            pos = 13
            for k in range(0, len(col_B)):
                true_where = where[pos: pos + 3][where[pos: pos + 3] >= 0]
                true_where_coarse = np.array([mapping.get(key) for key in true_where]).astype(int)

                _B[col_B[k], true_where_coarse[_B[col_B[k], true_where_coarse] == 0]] = np.nan
                _B[col_B[k], true_where_coarse] = np.nanmean(
                    (_B[col_B[k], true_where_coarse], pred[pos:pos + len(true_where_coarse)]), axis=0)

                pos = pos + 3

        return _B, d_neighs

    @staticmethod
    def pre_process(mass, d_neighs):
        cut_mass = lil_matrix(mass.shape)
        for i in range(0, mass.shape[0]):
            cut_mass[i, np.append(d_neighs[i], i)] = mass[i, np.append(d_neighs[i], i)]
        return cut_mass

    def define_hierarchy(self, levels = 2):
        # Here, we build the hierarchy, i.e, we construct the transfer operators
        if levels == 1:
            print("Coarse grid correction not required")
        else:
            mass = self.M
            l_hierarchy = [None] * (levels - 1)
            d_neighs = {}
            for i in range(0, levels-1):
                if d_neighs:
                    mass = self.pre_process(mass, d_neighs)
                C, F, C_neighs, F_neighs = self.coarsening(mass)
                mapp = self.map_coarse(C)
                patches, idx_fill = self.extract_patches(C, mass)
                patches = (patches - self.mean) / self.std
                res = self.model.predict(patches)
                B, d_neighs = self.fill_B(res, idx_fill, mass.shape[0], mapp, C)
                row_sums = B.sum(axis=1)
                Q = lil_matrix(B / row_sums[:, np.newaxis])
                del B
                del row_sums
                l_hierarchy[i] = Q
                mass = lil_matrix(Q.T @ mass @ Q)

            self.l_hierarchy = l_hierarchy




