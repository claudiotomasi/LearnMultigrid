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
        R = list(set(R)-set(neigh))

        rows = conn[neigh, :]
        for j in range(0, len(rows)):
            neigh_row = rows[j]
            FF_neighs[neigh[j]] = np.where(neigh_row > 0)[0]
        FF.extend(neigh)

    return CC, FF, CC_neighs, FF_neighs


def extract_patches(CC, _mat):
    # For each c_node I need to save the index where to fill B.
    # I need 'where' -> direct neighbours of c_node in fine grid
    # And 'where_nOfN -> direct neighbours of c_node in coarse grid
    #   it contains the neighbors of the neighbors of node
    #   If we intersect with c_index, we get the correct column indexes to fill row_B

    # Then I need 'intersecting' -> rows intersecting the column of B
    _patches = np.zeros((len(CC), 43))
    _fill_idxs = np.zeros((len(CC), 31), dtype = int)
    ii = 0
    mat = _mat.toarray()
    for c_node in CC:
        # patch consists of:
        # 1st is node_M - in pos 0
        # 2nd is row - in pos [1:7] (meaning until patch[6])
        # 3rd is neighs - pos [7:]
        # neighs consists of six blocks where:
        # 1st is neigh_node
        # 2nd row_neigh
        patch = np.ones(43) * -1
        fill_idx = np.ones(31) * -1
        # we need to use copy.copy(), otherwise we modify actual M
        row = copy.copy(mat[c_node, :])
        node_M = row[c_node]
        row[c_node] = 0
        where = np.where(row)[0]
        # print(where)
        row = row[row > 0]
        neighs = np.ones(36) * -1
        jj = 0
        B = []
        for neighbour in where:
            row_neigh = copy.copy(mat[neighbour, :])
            row_neigh[c_node] = 0
            B.extend(np.where(row_neigh)[0])
            neigh_node = row_neigh[neighbour]
            row_neigh[neighbour] = 0
            row_neigh = row_neigh[row_neigh > 0]
            neighs[jj] = neigh_node
            neighs[jj + 1: jj + len(row_neigh) + 1] = row_neigh
            jj = jj + 6

        # This remove duplicates maintaining original order
        B = sorted(set(B), key=B.index)
        # print(B)
        D = diff(diff(B, where), [c_node])
        # _auxset = set(D)
        where_nOfN = [x for x in D if x in CC]

        # print(where_nOfN)

        patch[0] = node_M
        patch[1: len(row) + 1] = row
        patch[7:] = neighs

        _patches[ii, :] = patch

        fill_idx[0] = c_node
        fill_idx[1:len(where) + 1] = where
        fill_idx[7: 7 + len(where_nOfN)] = where_nOfN

        # For each neighbour of c_node, i.e, for each index in 'where', we need 3 intersecting values
        # For c_node, its row was given by taking neighs(neighs(c_node)), minus 'where' minus 'c_node'
        # We need to do the same but for neighbour(c_node), thus for each element in 'where'
        # So, we start from nOfN which represents the neigh(where) and keep asking for its neighbours
        pos = 13
        for el in where:
            row_el = copy.copy(mat[el, :])
            row_el[c_node] = 0
            row_el[el] = 0
            row_neigh = np.where(row_el)[0]
            where_with_coarse = []
            for kk in row_neigh:
                neigh_row = copy.copy(mat[kk, :])
                where_with_coarse.extend(np.where(neigh_row)[0])
            where_with_coarse = sorted(set(where_with_coarse), key=where_with_coarse.index)
            where_with_coarse = [x for x in where_with_coarse if x in CC]
            where_with_coarse = diff(where_with_coarse, [c_node])
            fill_idx[pos: pos+len(where_with_coarse)] = where_with_coarse
            pos = pos + 3

        _fill_idxs[ii, :] = fill_idx

        ii = ii + 1
    return _patches, _fill_idxs


def fill_B(_res, _idx_fill, total_size):
    _B = np.zeros((total_size, len(_res)))
    for j in range(0, len(_res)):
        pred = _res[j]
        where = _idx_fill[j]

        node = where[0]
        col_B = where[1:7]
        col_B = col_B[col_B >= 0]
        row_B = where[7:13]
        row_B = row_B[row_B >= 0]
        # inters = where[13:]

        _B[node,node] = pred[0]
        _B[col_B,node] = pred[1:1 + len(col_B)]
        _B[node,row_B] = pred[7:7 + len(row_B)]

        pos = 13
        for k in range(0, len(col_B)):
            true_where = where[pos: pos+3][where[pos: pos+3] >= 0]
            _B[col_B[k], true_where] = pred[pos:pos+len(true_where)]
            pos = pos + 3

    return _B

# put 300 for 10 its on 12k dofs
# mesh = Mesh2D(1000)
# mesh.refine(regular = True)



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path_model = path = '../data/models/'

model = tf.keras.models.load_model(path + '2D_12_KH.h5', compile=False)


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

#
timings = []
dim = []
for kk in range(1, 21):
    ne = 16 + 192 * (kk - 1)
    mesh = Mesh2D(ne)
    mesh.refine(regular=False)
    dim.append(mesh.get_np())
    q = Quadrature2D(3)
    phi = FunctionTriangle(1)
    fun = LoadFunction(f)

    mass = MassMatrix(mesh)

    M = mass.compute_mass_2d(phi, q)

    C, F, C_neighs, F_neighs = coarsening(M)

    patches, idx_fill = extract_patches(C, M)

    tot = time.time()

    start = time.time()
    embed = mesh.embedding()
    end = time.time()
    timeEmbed = end - start
    print("Time for assembly mass embedded mesh: ", timeEmbed)
    # embed.plot_mesh()

    start = time.time()
    embed_mass = MassMatrix(embed)
    M_2 = embed_mass.compute_mass_2d(phi, q)
    end = time.time()
    timeMassEmbed = end - start
    print("Time for assembly mass embedded mesh: ", timeMassEmbed)

    start = time.time()
    patches_2, idx_fill_2 = extract_patches(C, M_2)
    end = time.time()
    timeExtractPatchesEmbed = end - start
    print("Time for extracting patches embedded mass: ", timeExtractPatchesEmbed)

    start = time.time()
    for i in range(0, patches.shape[0]):
        for j in range(0, patches.shape[1]):
            if patches[i, j] == -1:
                patches[i, j] = patches_2[i, j]
    end = time.time()
    timeCopyingNewPatch = end - start
    print("Time for copying new added node patches into old patches: ", timeCopyingNewPatch)

    start = time.time()
    patches = (patches - mean) / std
    res = model.predict(patches)
    end = time.time()
    timeNormAndPred = end - start
    print("Time normalize patches and NN prediciton: ", timeNormAndPred)
    timeComputing = time.time() - tot

    # start = time.time()
    # B = fill_B(res, idx_fill, M.shape[0])
    # row_sums = B.sum(axis=1)
    # Q = B / row_sums[:, np.newaxis]
    # end = time.time()
    # timeComputeQ = end - start
    # print("Time computing Q: ", timeComputeQ)

    print("Time total: ", timeComputing)
    timings.append(timeComputing - timeEmbed - timeExtractPatchesEmbed)

path_matlab = "../data/from_matlab/"
problem = path_matlab + "times_for_py.mat"
pp = scipy.io.loadmat(problem)
times = pp['times']
times = times.flatten().tolist()
problem = path_matlab + "times_for_py_2.mat"
pp = scipy.io.loadmat(problem)
times.extend(pp['times'].flatten())

# xx = np.arange(1, kk + 1) * 16
scale = 'log'
plt.plot(dim, timings, label='Through Predictions')
plt.yscale(scale)
plt.legend()
plt.title("Computation Time")
plt.ylabel('time')
plt.xlabel('dimension')

plt.plot(dim, times, label='Through Intersections')
plt.yscale(scale)
plt.legend()
plt.title("Assembly CPU Time")
plt.ylabel('time')
plt.xlabel('dimension')
plt.savefig('../data/timings.png', dpi=300)
plt.show()

# # TODO: CHANGE, I NEED TO USE LIL-MATRIX NOW. THEN USE A = B[:] \
# #  TO COPY THE CONTENTS AND NOT THE REFERENCE \
# #  ALSO LOOK TO HOW THE ROW ELEMENTS ARE EXTRACTED AND USE lil_matrix METHODS
#
#
