from learn_multigrid.mesh.Mesh2D import *
from learn_multigrid.assembly.StiffnessMatrix import *
from learn_multigrid.assembly.LoadVector import *
from learn_multigrid.assembly.LoadFunction import *
from learn_multigrid.assembly.MassMatrix import *
from learn_multigrid.assembly.Quadrature import *
from learn_multigrid.assembly.ShapeFunction import *
import copy
import random
from learn_multigrid.solvers.Multigrid import *
import tensorflow as tf
import os
import time
import scipy.io
from scipy.sparse import lil_matrix
import pyamg


def f(x):
    # return np.sin(2 * np.pi * x)
    return -1


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def get_conn(mat):
    matrix = mat[:]
    matrix.setdiag(0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                matrix[i, j] = 1
    return matrix.toarray()


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


def scaling_vnodes(_neighs):
    switcher = {
        2: [6, 1],
        3: [3, 2],
        4: [2, 3],
        5: [1.5, 4]
    }
    return switcher.get(_neighs, "Invalid month")


def create_virtual_nodes(row, node_M):
    actual_neigh = len(row)
    virtual_neighs = []
    if actual_neigh < 6:
        # create virtual neighbours
        scale = scaling_vnodes(actual_neigh)
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


def direct_neighs(where, mat, c_node):
    B = []
    neighs = np.ones(36) * -1
    jj = 0
    if c_node == 569:
        print("ciaoo")

    for neighbour in where:
        row_neigh = copy.copy(mat[neighbour, :])
        row_neigh[c_node] = 0
        B.extend(np.where(row_neigh)[0])
        neigh_node = row_neigh[neighbour]
        row_neigh[neighbour] = 0
        row_neigh = row_neigh[row_neigh > 0]

        # Fill missing entries in existing nodes
        actual_neigh = len(row_neigh)
        if actual_neigh > 5:
            while actual_neigh > 5:
                pos = np.where(row_neigh == np.min(row_neigh))[0][0]
                row_neigh = np.delete(row_neigh, [pos])
                actual_neigh = len(row_neigh)
        # 5 because we erase row[c_node]
        if actual_neigh < 5:
            # create virtual neighbours
            # +1 because we delete the c_node value, but we still need to consider it
            scale = scaling_vnodes(actual_neigh + 1)
            up = scale[0]
            down = scale[1]
            row_neigh = np.append(row_neigh, np.ones((1, 5 - actual_neigh)) * neigh_node / down)
        neighs[jj] = neigh_node
        neighs[jj + 1: jj + len(row_neigh) + 1] = row_neigh
        jj = jj + 6
    return B, neighs


def intersecting_rows(where, mat, c_node, CC, fill_idx):
    pos = 13
    for el in where:
        row_el = copy.copy(mat[el, :])
        row_el[c_node] = 0
        row_el[el] = 0
        row_neigh = np.where(row_el)[0]
        row_el = row_el[row_el > 0]
        actual_neigh = len(row_neigh)
        if actual_neigh > 5:
            while actual_neigh > 5:
                pos_to_del = np.where(row_el == np.min(row_el))[0][0]
                row_el = np.delete(row_el, [pos_to_del])
                row_neigh = np.delete(row_neigh, [pos_to_del])
                actual_neigh = len(row_el)

        where_with_coarse = []
        for kk in row_neigh:
            neigh_row = copy.copy(mat[kk, :])
            wh = np.where(neigh_row)[0]
            neigh_row = neigh_row[neigh_row > 0]
            actual_neigh = len(neigh_row)
            # Since we do not erase nodes we need to consider all 7 entries
            if actual_neigh > 7:
                while actual_neigh > 7:
                    pos_to_del = np.where(neigh_row == np.min(neigh_row))[0][0]
                    neigh_row = np.delete(neigh_row, [pos_to_del])
                    wh = np.delete(wh, [pos_to_del])
                    actual_neigh = len(neigh_row)
            where_with_coarse.extend(wh)

        where_with_coarse = sorted(set(where_with_coarse), key=where_with_coarse.index)
        # st = time.time()
        w_w_c = set(where_with_coarse)
        set_cc = set(CC)
        where_with_coarse = sorted(w_w_c & set_cc, key=where_with_coarse.index)

        where_with_coarse = diff(where_with_coarse, [c_node])
        while len(where_with_coarse) > 3:
            # pick = random.randint(0, len(where_with_coarse) - 1)
            pick = 0
            where_with_coarse = np.delete(where_with_coarse, [pick])

        fill_idx[pos: pos + len(where_with_coarse)] = where_with_coarse
        pos = pos + 3
    return fill_idx


def single_extraction(row, where, mat, c_node, CC, node_M):
    patch = np.ones(43) * -1
    fill_idx = np.ones(31) * -1

    st = time.time()
    B, neighs = direct_neighs(where, mat, c_node)
    print("direct_neigh: ", time.time() - st)
    # This remove duplicates maintaining original order
    B = sorted(set(B), key=B.index)
    # print(B)
    D = diff(diff(B, where), [c_node])
    # _auxset = set(D)
    set_D = set(D)
    set_cc = set(CC)
    where_nOfN = sorted(set_D & set_cc, key=D.index)

    # print(where_nOfN)

    patch[0] = node_M

    virtual_neighs, row = create_virtual_nodes(row, node_M)

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
    fill_idx = intersecting_rows(where, mat, c_node, CC, fill_idx)
    return patch, fill_idx


def extract_patches(CC, _mat):
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
    mat = _mat.toarray()

    to_add = 0
    for c_node in CC:
        row = copy.copy(mat[c_node, :])
        row[c_node] = 0
        row = row[row > 0]
        actual_neigh = len(row)
        additional = (actual_neigh - 6)
        if additional > 0:
            to_add = to_add + additional

    for c_node in CC:
        st = time.time()
        # patch consists of:
        # 1st is node_M - in pos 0
        # 2nd is row - in pos [1:7] (meaning until patch[6])
        # 3rd is neighs - pos [7:]
        # neighs consists of six blocks where:
        # 1st is neigh_node
        # 2nd row_neigh
        if c_node == 569:
            print('ciaop')
        # patch = np.ones(43) * -1
        # fill_idx = np.ones(31) * -1
        # we need to use copy.copy(), otherwise we modify actual M
        row = copy.copy(mat[c_node, :])
        node_M = row[c_node]
        row[c_node] = 0
        where = np.where(row)[0]
        # print(where)
        row = row[row > 0]

        actual_neigh = len(row)
        additional = actual_neigh - 6
        row_orig = copy.copy(row)
        where_origin = copy.copy(where)
        # while actual_neigh > 6:

        # Works only if neighs are 6 + 1
        if additional > 0:
            ordered = np.argsort(row_orig)
            add_patches = np.zeros((additional+1, 43))
            add_fill_idxs = np.zeros((additional+1, 31), dtype=int)
            for add in range(0, additional+1):
                # pos = np.where(row_orig == np.min(row_orig))[0][0]
                pos = ordered[add:add+additional]
                row = np.delete(row_orig, [pos])
                where = np.delete(where_origin, [pos])
                # actual_neigh = len(row)

                patch_add, fill_idx_add = single_extraction(row, where, mat, c_node, CC, node_M)
                add_patches[add, :] = patch_add
                add_fill_idxs[add, :] = fill_idx_add
            _patches[ii, :] = add_patches[0, :]
            _fill_idxs[ii, :] = add_fill_idxs[0, :]
            addition_patches = np.vstack((addition_patches, add_patches[1:, :]))
            addition_idx = np.vstack((addition_idx, add_fill_idxs[1:, :]))

        else:
            patch, fill_idx = single_extraction(row, where, mat, c_node, CC, node_M)
            _patches[ii, :] = patch
            _fill_idxs[ii, :] = fill_idx

        print("additional: ", additional, "coarse node: ", c_node, " on total: ", mat.shape[0],
              "time: ", time.time() - st)
        ii = ii + 1
    _patches = np.vstack((_patches, addition_patches[1:, :]))
    _fill_idxs = np.vstack((_fill_idxs, addition_idx[1:, :]))
    return _patches, _fill_idxs


def map_coarse(_C):
    dim = len(_C)
    enum = list(range(0, dim))
    _mapp = dict(zip(_C, enum))
    return _mapp


def fill_B(_res, _idx_fill, total_size, mapping, CC):
    _B = np.zeros((total_size, len(CC)))
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

        # inters = where[13:]
        if j == len(_res) - 2:
            print("ciao")
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

    return _B

np.random.seed(42)
path_matlab = "../data/from_matlab/"
problem = path_matlab + "3d_poisson.mat"
pp = scipy.io.loadmat(problem)
A = lil_matrix(pp['A'])
M = lil_matrix(pp['M'])
rhs = pp['rhs']


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path_model = path = '/Users/claudio/Desktop/Testing_NNs/2D/only_linear/ne_16_16K_plus208_500foreach/'
#
model = tf.keras.models.load_model(path + '2D_best_51.h5', compile=False)
#
std = np.array([0.00096162, 0.00017077, 0.00017155, 0.00017346, 0.00017494,
       0.00017445, 0.00017545, 0.00089226, 0.0001737 , 0.00015616,
       0.00015555, 0.00015595, 0.00015473, 0.00089322, 0.00017131,
       0.00015788, 0.00015447, 0.00015527, 0.00015554, 0.00089408,
       0.00017356, 0.00015394, 0.00015703, 0.00015635, 0.00015642,
       0.00089457, 0.00017565, 0.00015473, 0.00015491, 0.00015655,
       0.00015764, 0.00089533, 0.00016921, 0.00015635, 0.00015433,
       0.00015793, 0.00015559, 0.00089575, 0.00016917, 0.00015764,
       0.00015793, 0.00015269, 0.00015533])

mean = np.array([1.40203540e-04, 2.30680698e-05, 2.29900874e-05, 2.32328129e-05,
       2.35415238e-05, 2.35701587e-05, 2.38008871e-05, 1.38469732e-04,
       2.34355592e-05, 2.30843163e-05, 2.29589663e-05, 2.30824799e-05,
       2.28403406e-05, 1.38596834e-04, 2.34634341e-05, 2.33283670e-05,
       2.28771969e-05, 2.29351001e-05, 2.30026484e-05, 1.38834346e-04,
       2.34474049e-05, 2.27796488e-05, 2.31515569e-05, 2.30937517e-05,
       2.31291710e-05, 1.38891966e-04, 2.33384001e-05, 2.28403406e-05,
       2.28333904e-05, 2.30710686e-05, 2.32672421e-05, 1.38822459e-04,
       2.29419931e-05, 2.30937517e-05, 2.28604871e-05, 2.33754848e-05,
       2.29805838e-05, 1.38766429e-04, 2.27404875e-05, 2.32672421e-05,
       2.33754848e-05, 2.26306232e-05, 2.29517041e-05])


# C, F, C_neighs, F_neighs = coarsening(M)
# np.save(path_data + "coarse.npy", C)

# path_data = "/Users/claudio/Desktop/PhD/Tesi/dati_test/smile50/"
#
# C = np.load(path_data + "coarse.npy")
# mapp = map_coarse(C)
#
# patches = np.load(path_data + "patch.npy")
# idx_fill = np.load(path_data + "idx_fill.npy")
#
# path_for_saving = path_data + "ne_16_10K_plus208_1000foreach/"
# #
# patches = (patches - mean) / std
# res = model.predict(patches)
# print("Done predictions")

# start = time.time()
# B = fill_B(res, idx_fill, M.shape[0], mapp, C)
# end = time.time()
# print("Done filling B in ", (end - start))
# row_sums = B.sum(axis=1)
# Q = B / row_sums[:, np.newaxis]
#
# nmg = SemiGeometricMG(A, rhs, Q)
# nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-09, max_iterations=20)
# print("Done filling B full in ", (end - start))
nmg = NeuralMG_2D(A, rhs,  model, M, std, mean)
nmg.define_hierarchy(levels = 2)
nmg.solve(smoother="GaussSeidel", smooth_steps=3, error=1e-09, max_iterations=40)
nn_res = nmg.get_track_res()

# np.save(path_for_saving + "res_NMG-2D_best_51_smile50.npy", nn_res)

ml = pyamg.ruge_stuben_solver(A, max_levels=2, presmoother=[('gauss_seidel', {'iterations': 3})], coarse_solver='splu')
b = rhs.flatten()
amg_res = []
x = ml.solve(b, tol=1e-09, residuals = amg_res, cycle = 'V')
amg_res[0] = nn_res[0]
lesser = np.where(np.array(amg_res)<1e-09)[0]
amg_res = amg_res[:lesser[0]+1]

x_scale = range(0,len(amg_res),2)
plt.close()
plt.rcParams['figure.figsize']=[10, 9]
plt.yscale("log")
plt.plot(nn_res,  'o-', lw=1, ms = 5, label='Neural Multigrid')
plt.plot(amg_res,  's-', color = "lawngreen", lw=1, ms = 5, label='AMG')
plt.xlabel('iterations', fontsize=15)
plt.ylabel('residual', fontsize=15)
plt.grid(True, linestyle='--', color='k', alpha=0.3, lw = 1)
plt.title("Residual decreasing", fontsize = 28, pad = 10)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=20)
plt.xticks(x_scale)
# plt.savefig(path_for_saving+"NMG_vs_AMG", dpi=300)
plt.show()



