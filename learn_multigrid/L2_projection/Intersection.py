import numpy as np


class Intersection:

    def __init__(self, fine_mesh, coarse_mesh):
        self.fine_mesh = fine_mesh
        self.coarse_mesh = coarse_mesh
        self.intersections = None
        self.int_coord = None
        self.union = None

    def get_info(self):
        return self.intersections, self.int_coord, self.union

    def get_intersections(self):
        return self.intersections

    def find_intersections2d(self):
        fine_mesh = self.fine_mesh
        coarse_mesh = self.coarse_mesh

        p = fine_mesh.get_points()
        conn = fine_mesh.get_connections()

        c_p = coarse_mesh.get_points()
        c_conn = coarse_mesh.get_connections()

        intersected = np.zeros((conn.shape[0],2), dtype=int)
        for i in range(0, c_conn.shape[0]):
            fine = np.linspace(i*4, i*4+3, 4, dtype = int)
            intersected[fine,0] = fine
            intersected[fine,1] = i
        self.intersections = intersected


    def find_intersections1d(self):
        print("Finding intersections...")
        fine_mesh = self.fine_mesh
        coarse_mesh = self.coarse_mesh
        """

        :type coarse_mesh: Mesh1D
        :type fine_mesh: Mesh1D

        intersections: contains (fine_el, coarse_el) which intersect
        int_coord: contains the coordinates of the points (of segment) of each intersection
        """
        intersections = np.ndarray(shape=(0, 2), dtype = int)
        int_coord = np.ndarray(shape=(0, 2), dtype = float)

        conn = fine_mesh.get_connections()
        coarse_conn = coarse_mesh.get_connections()
        fine_x = fine_mesh.get_connections()
        coarse_x = coarse_mesh.get_connections()

        union = np.union1d(fine_x, coarse_x)

        k = 0
        for i in range(0, fine_mesh.get_ne()):
            for j in range(0, coarse_mesh.get_ne()):
                left = conn[i, 0]
                right = conn[i, 1]

                c_left = coarse_conn[j, 0]
                c_right = coarse_conn[j, 1]

                if left >= c_right:
                    continue
                else:
                    if right <= c_left:
                        continue
                    else:
                        intersections = np.vstack((intersections, [i, j]))
                        int_coord = np.vstack((int_coord, [union[k], union[k+1]]))
                        k += 1

        self.intersections = intersections
        self.int_coord = int_coord
        self.union = union






