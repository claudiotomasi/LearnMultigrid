import scipy.io
from learn_multigrid.mesh.Mesh2D import *
from learn_multigrid.assembly.StiffnessMatrix import *
from learn_multigrid.assembly.LoadVector import *
from learn_multigrid.assembly.LoadFunction import *
from learn_multigrid.assembly.MassMatrix import *
from learn_multigrid.assembly.Quadrature import *
from learn_multigrid.assembly.ShapeFunction import *
import copy
from learn_multigrid.solvers.Multigrid import *
import tensorflow as tf
import os
import time
import scipy.io
from scipy.sparse import lil_matrix

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
    row = row.toarray()[0]
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
        tt = time.time()
        row_neigh = copy.copy(mat[neighbour, :])
        row_neigh[0, c_node] = 0
        B.extend(row_neigh.nonzero()[1])
        # print("TTT: ", time.time() - tt)
        neigh_node = row_neigh[0, neighbour]
        row_neigh[0,neighbour] = 0
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
        row_el[0, c_node] = 0
        row_el[0, el] = 0
        row_neigh = row_el.nonzero()[1]
        row_el = row_el[row_el > 0]
        actual_neigh = len(row_neigh)
        if actual_neigh > 5:
            while actual_neigh > 5:
                pos_to_del = np.where(row_el == np.min(row_el))[0][0]
                row_el = np.delete(row_el, [pos_to_del])
                row_neigh = np.delete(row_neigh, [pos_to_del])
                actual_neigh = row_el.getnnz()

        where_with_coarse = []
        for kk in row_neigh:
            neigh_row = copy.copy(mat[kk, :])
            wh = neigh_row.nonzero()[1]
            neigh_row = neigh_row[neigh_row > 0]
            actual_neigh = neigh_row.getnnz()
            # Since we do not erase nodes we need to consider all 7 entries
            if actual_neigh > 7:
                while actual_neigh > 7:
                    pos_to_del = np.where(neigh_row == np.min(neigh_row))[0][0]
                    neigh_row = np.delete(neigh_row, [pos_to_del])
                    wh = np.delete(wh, [pos_to_del])
                    actual_neigh = neigh_row.getnnz()
            where_with_coarse.extend(wh)

        where_with_coarse = sorted(set(where_with_coarse), key=where_with_coarse.index)
        # st = time.time()
        w_w_c = set(where_with_coarse)
        set_cc = set(CC)
        where_with_coarse = sorted(w_w_c & set_cc, key=where_with_coarse.index)

        # where_with_coarse = [x for x in where_with_coarse if x in CC]
        # print("TIME intersecting rows: ", time.time() - st)
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
    st_diff = time.time()
    D = diff(diff(B, where), [c_node])
    # _auxset = set(D)
    # where_nOfN = [x for x in D if x in CC]
    set_D = set(D)
    set_cc = set(CC)
    where_nOfN = sorted(set_D & set_cc, key=D.index)
    print("time sigle_extr intersection: ", time.time() - st_diff)

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
    st = time.time()
    fill_idx = intersecting_rows(where, mat, c_node, CC, fill_idx)
    print("intersect coarse: ", time.time() - st)
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

    for c_node in CC:

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
            st_single = time.time()
            patch, fill_idx = single_extraction(row, where, mat, c_node, CC, node_M)
            print("time single: ", time.time()-st_single)
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


path_matlab = "../data/from_matlab/"
problem = path_matlab + "small_problem.mat"
pp = scipy.io.loadmat(problem)

A = lil_matrix(pp['A'])
M = lil_matrix(pp['M'])
rhs = pp['rhs']
L2 = pp['Q']
path_model = path = '../data/models/'
#
model = tf.keras.models.load_model(path + '2D_12_KH.h5', compile=False)
#
std = np.array([6.46081319e-04, 6.56278542e-05, 1.44380665e-04, 2.01278225e-04,
                4.05887607e-05, 1.44852542e-04, 6.59623314e-05, 5.60079604e-04,
                1.31119065e-04, 1.35756622e-04, 9.76726150e-05, 8.71917170e-05,
                6.30029006e-05, 5.97089104e-04, 6.90149936e-05, 7.54854156e-05,
                9.76312402e-05, 1.11220354e-04, 1.26176708e-04, 6.50233509e-04,
                4.88818477e-05, 9.15482680e-05, 1.29714979e-04, 1.41831294e-04,
                6.33190965e-05, 6.37140535e-04, 2.01655903e-04, 6.30029006e-05,
                1.41933771e-04, 1.41892169e-04, 6.32955373e-05, 5.74263180e-04,
                6.64051795e-05, 1.41831294e-04, 8.72306803e-05, 8.72795778e-05,
                6.35418401e-05, 5.73858149e-04, 1.44794401e-04, 6.32955373e-05,
                8.72795778e-05, 8.71791125e-05, 1.41874219e-04])

mean = np.array([2.00689432e-04, 1.90138926e-05, 4.44341181e-05, 6.22542011e-05,
                 1.14234973e-05, 4.44871006e-05, 1.90766223e-05, 1.78162953e-04,
                 4.18346546e-05, 4.33776652e-05, 2.82244908e-05, 2.66899185e-05,
                 1.90223309e-05, 1.84076492e-04, 1.93429694e-05, 2.05190563e-05,
                 2.82222663e-05, 3.00061370e-05, 4.15519444e-05, 2.02526276e-04,
                 1.23227822e-05, 2.19915157e-05, 4.24219182e-05, 4.44695701e-05,
                 1.90662885e-05, 2.00735758e-04, 6.22890683e-05, 1.90223309e-05,
                 4.44762675e-05, 4.44681140e-05, 1.90564797e-05, 1.80504669e-04,
                 1.90736689e-05, 4.44695701e-05, 2.66888763e-05, 2.67090956e-05,
                 1.90763577e-05, 1.80483765e-04, 4.44763070e-05, 1.90564797e-05,
                 2.67090956e-05, 2.66949169e-05, 4.44703430e-05])
print("a")
C, F, C_neighs, F_neighs = coarsening(M)
print("Done coarsening")
mapp = map_coarse(C)

patches, idx_fill = extract_patches(C, M)
print("Done extraction of patches")
patches = (patches - mean) / std
res = model.predict(patches)
print("Done predictions")

B = fill_B(res, idx_fill, M.shape[0], mapp,C)
row_sums = B.sum(axis=1)
Q = B / row_sums[:, np.newaxis]


semi = SemiGeometricMG(A, rhs, L2)
semi.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-09, max_iterations=40)
res_sgmg = semi.get_track_res()

nmg = SemiGeometricMG(A, rhs, Q)
nmg.solve(levels=2, smoother="GaussSeidel", smooth_steps=3, error=1e-09, max_iterations=40)
res_nmg = nmg.get_track_res()

plt.close()
plt.rcParams['figure.figsize']=[10, 10]
plt.yscale("log")
# fig, a = plt.subplots(1, 1)
# fig.subplots_adjust(hspace=0.25, wspace=0.25)
# plt.figure()
plt.plot(res_nmg,  'o-', lw=1, ms = 5, label='Neural Multigrid')
plt.xlabel('iterations', fontsize=26)
plt.ylabel('residual', fontsize=26)
plt.plot(res_sgmg , 's-', lw=1, ms = 5, label='SemiGM Multigrid',)
plt.grid(True, linestyle='--', color='k', alpha=0.3, lw = 1)
plt.title("Residual decreasing", fontsize = 35, pad = 12)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=20)
# plt.savefig(path + '/new_res_un.png', dpi=300)
plt.show()