from numpy import array, zeros
from random import random


class Matrix(object):
    def __init__(self, rows: int = 0, cols: int = 0, randoms: bool = False):
        self.rows, self.cols, self.matrix = rows, cols, []
        if rows != 0 and cols != 0:
            if randoms:
                self.matrix = array([[random() for j in range(0, cols)] for i in range(0, rows)], dtype=object)
            else:
                self.matrix = zeros([rows, cols], dtype=float)

    def load_list(self, value: any):
        if isinstance(value[0], list):
            self.rows, self.cols = len(value), len(value[0])
            self.matrix = [[value[i][j] for j in range(self.cols)] for i in range(self.rows)]
        else:
            self.rows, self.cols = len(value), 1
            self.matrix = [[value[i]] for i in range(self.rows)]
        self.matrix = array(self.matrix, dtype=object)
        return self

    def __getitem__(self, key: int):
        return self.matrix[key]

    def __setitem__(self, key: int, value: any):
        self.matrix[key] = value

    def __str__(self):
        return str(self.matrix)
