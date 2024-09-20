# TODO: implement gradient and other orders

class LoadFunction:

    def __init__(self, fun):
        self.fun = fun

    def evaluate(self, points = None):
        result = self.fun(points)

        return result

    def get_functions(self):
        return self.fun
