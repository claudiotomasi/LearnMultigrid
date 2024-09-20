import matplotlib.pyplot as plt
import numpy as np


def plot_intersections(fine_mesh, coarse_mesh, union):
    fine_x = fine_mesh.get_mesh()
    coarse_x = coarse_mesh.get_mesh()
    fine_y = np.zeros(len(fine_x))
    coarse_y = np.zeros(len(coarse_x))

    union_y = np.zeros(len(union))

    plt.subplot(3, 1, 1)
    plt.plot(fine_x, fine_y, 'ro')
    plt.grid()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('Fine Mesh')

    plt.subplot(3, 1, 2)
    plt.plot(coarse_x, coarse_y, 'ro')
    plt.grid()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('Coarse Mesh')

    plt.subplot(3, 1, 3)
    plt.plot(union, union_y, 'ro')
    plt.grid()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('Intersections')

    plt.show()
