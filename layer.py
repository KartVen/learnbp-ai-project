from numpy import array, zeros, random


class Layer(object):
    def __init__(self, rows=0, cols=0, randoms=False):
        self.rows, self.cols, self.matrix = rows, cols, []
        if randoms:
            self.matrix = array([random.rand(cols) for row in range(0, rows)])
        else:
            self.matrix = zeros([rows, cols], dtype=float)

    def load_list(self, obj: list):
        self.rows, self.cols = len(obj), 1
        self.matrix = array([[value] for value in obj])
        return self

    def load_array(self, obj: array):
        self.rows, self.cols = len(obj), len(obj[0])
        self.matrix = obj
        return self

    def sum_cols(self):
        self.cols = 1
        self.matrix = [[sum(self.matrix[row])] for row in range(0, self.rows)]

    def transpose(self):
        self.load_array(self.matrix.T)
        return self

    @property
    def T(self):
        self.load_array(self.matrix.T)
        return self

    def __str__(self):
        return str(self.matrix)

    def __mul__(self, other: any):
        if type(other) not in [Layer, float, int]:
            raise TypeError("Supported operand type(s) for *: Layer, int, float")
        self_copy, self_copy.rows, self_copy.cols = Layer(), self.rows, self.cols
        if type(other) == Layer:
            if self.rows == other.rows and self.cols == other.cols:
                self_copy.matrix = [[self.matrix[row][col] * other.matrix[row][col] for col in range(0, self.cols)]
                                    for row in range(0, self.rows)]
            elif other.rows == 1 and self.cols == other.cols:
                self_copy.matrix = [[self.matrix[row][col] * other.matrix[0][col] for col in range(0, self.cols)]
                                    for row in range(0, self.rows)]
            else:
                raise ValueError(f"Shapes ({self.rows},{self.cols}) and ({other.rows},{other.cols}) not aligned")
        else:
            self_copy.matrix = [[self.matrix[row][col] * other for col in range(0, self.cols)]
                                for row in range(0, self_copy.rows)]
        return self_copy

    def __add__(self, other: any):
        if type(other) not in [Layer, float, int]:
            raise TypeError("Supported operand type(s) for +: Layer, int, float")
        temp_layer, temp_layer.rows, temp_layer.cols = Layer(), self.rows, self.cols
        if type(other) == Layer:
            if self.rows == other.rows and self.cols == other.cols:
                temp_layer.matrix = [[self.matrix[row][col] + other.matrix[row][col] for col in range(0, self.cols)]
                                     for row in range(0, self.rows)]
            elif other.rows == 1 and self.cols == other.cols:
                temp_layer.matrix = [[self.matrix[row][col] + other.matrix[0][col] for col in range(0, self.cols)]
                                     for row in range(0, self.rows)]
            else:
                raise ValueError(f"Shapes ({self.rows},{self.cols}) and ({other.rows},{other.cols}) not aligned")
        else:
            temp_layer.matrix = [[self.matrix[row][col] + other for col in range(0, self.cols)]
                                 for row in range(0, temp_layer.rows)]
        return temp_layer

    def __sub__(self, other: any):
        if type(other) not in [Layer, float, int]:
            raise TypeError("Supported operand type(s) for -: Layer, int, float")
        temp_layer, temp_layer.rows, temp_layer.cols = Layer(), self.rows, self.cols
        if type(other) == Layer:
            if self.rows == other.rows and self.cols == other.cols:
                temp_layer.matrix = [[self.matrix[row][col] - other.matrix[row][col] for col in range(0, self.cols)]
                                     for row in range(0, self.rows)]
            elif other.rows == 1 and self.cols == other.cols:
                temp_layer.matrix = [[self.matrix[row][col] - other.matrix[0][col] for col in range(0, self.cols)]
                                     for row in range(0, self.rows)]
            else:
                raise ValueError(f"Shapes ({self.rows},{self.cols}) and ({other.rows},{other.cols}) not aligned")
        else:
            temp_layer.matrix = [[self.matrix[row][col] - other for col in range(0, self.cols)]
                                 for row in range(0, temp_layer.rows)]
        return temp_layer
