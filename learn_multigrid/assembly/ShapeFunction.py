# TODO: implement gradient and other orders
from abc import ABC, abstractmethod
import numpy as np


class ShapeFunction(ABC):

    def __init__(self, order):
        self.order = order
        self.phi = None

    def evaluate(self, points = None, index = None):
        result = np.ndarray(shape=(0, points.size), dtype=float)
        if index is None:
            # 1st row: evaluation of phi1 in all the points; same for phi2 in 2nd row
            for f in self.phi:
                result = np.vstack((result, f(points)))
        else:
            if index in range(0, self.phi.size):
                result = np.ndarray(shape=(0, points.size), dtype=float)
                f = self.phi[index]
                result = f(points)
            else:
                print("No shape function available")
                result = -100
        return result

    def get_functions(self):
        return self.phi


class Function(ShapeFunction):
    def __init__(self, order):
        super().__init__(order)
        self.phi = self.order_to_function(order)

    @staticmethod
    def order_to_function(order):
        switcher = {
            2: np.array([lambda x: 1 - x, lambda x: x]),
        }
        return switcher.get(order, "Invalid order")


class Gradient(ShapeFunction):

    def __init__(self, order):
        super().__init__(order)
        self.phi = self.order_to_gradient(order)

    @staticmethod
    def order_to_gradient(order):
        switcher = {
            2: np.array([lambda x: -1, lambda x: 1]),
        }
        return switcher.get(order, "Invalid order")


class FunctionTriangle(ShapeFunction):
    def __init__(self, order):
        super().__init__(order)
        self.phi = self.order_to_function(order)

    @staticmethod
    def order_to_function(order):
        switcher = {
            1: np.array([lambda p: 1 - p[0] - p[1],
                         lambda p: p[0],
                         lambda p: p[1]]),
        }
        return switcher.get(order, "Invalid order")


class GradientTriangle(ShapeFunction):
    def __init__(self, order):
        super().__init__(order)
        self.phi = self.order_to_gradient(order)

    @staticmethod
    def order_to_gradient(order):
        switcher = {
            1: np.array([lambda p: np.array([np.array([-1, -1])]).T,
                         lambda p: np.array([np.array([1, 0])]).T,
                         lambda p: np.array([np.array([0, 1])]).T]),
        }
        return switcher.get(order, "Invalid order")