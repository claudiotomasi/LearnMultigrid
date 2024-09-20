# TODO: put here different points and weights
import numpy as np


class Quadrature:
    # p = np.array([0.11270166537926, 0.50000000000000, 0.88729833462074])
    # w = np.array([0.27777777777778, 0.44444444444444, 0.27777777777778])

    def __init__(self, order):
        self.p = self.order_to_points(order)
        self.w = self.order_to_weights(order)

    def get_points(self):
        return self.p

    def get_weights(self):
        return self.w

    def compute(self, phi, index):
        p = self.get_points()
        res = 0
        i = index[0]
        j = index[1]
        for k in range(0, len(p)):
            res += phi.evaluate(p[k], i) * phi.evaluate(p[k], j) * self.w[k]
        return res

    # def compute_inter(self, f1, f2, fine_p, coarse_p):
    #     result = 0
    #     for i in range(0, len(fine_p)):
    #         result += f1(fine_p[i]) * f2(coarse_p[i]) * self.w[i]
    #     return result

    def compute_inter(self, phi, index, fine_p, coarse_p):
        result = 0
        i = index[0]
        j = index[1]
        for k in range(0, len(fine_p)):
            result += phi.evaluate(fine_p[k], i) * phi.evaluate(coarse_p[k], j) * self.w[k]
        return result

    def compute_single(self, phi, index, fun):
        p = self.get_points()
        res = 0
        i = index
        for k in range(0, len(p)):
            res += phi.evaluate(p[k], i) * fun.evaluate(p[k]) * self.w[k]
        return res

    @staticmethod
    def order_to_points(order):
        switcher = {
            1: np.array([0]),
            3: np.array([0.11270166537926, 0.50000000000000, 0.88729833462074]),
        }
        return switcher.get(order, "Invalid order")

    @staticmethod
    def order_to_weights(order):
        switcher = {
            1: np.array([2]),
            3: np.array([0.27777777777778, 0.44444444444444, 0.27777777777778]),
        }
        return switcher.get(order, "Invalid order")


class Quadrature2D(Quadrature):

    def __init__(self, order):
        super().__init__(order)

    def compute_grad(self, d_phi, jac_inv, index, ):
        p = self.get_points()
        w = self.get_weights()
        res = 0
        i = index[0]
        j = index[1]
        for k in range(0, len(p)):

            res += ((jac_inv @ d_phi.evaluate(p[k], i)).T @ jac_inv @ d_phi.evaluate(p[k], j))[0][0] * w[i]

        return res

    @staticmethod
    def order_to_points(order):
        switcher = {
            3: np.array([[0.16666666666667, 0.16666666666667],
                        [0.16666666666667, 0.66666666666667],
                        [0.66666666666667, 0.16666666666667]]),
        }
        return switcher.get(order, "Invalid order")

    @staticmethod
    def order_to_weights(order):
        switcher = {
            3: np.array([1/6, 1/6, 1/6]),
        }
        return switcher.get(order, "Invalid order")

