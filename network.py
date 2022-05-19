from numpy import random, exp, array
from layer import Layer
from tabulate import tabulate


class Network(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.__input_size, self.__hidden_size, self.__output_size = input_size, hidden_size, output_size
        self.__weights = array(
            [Layer(hidden_size, input_size, True), Layer(output_size, hidden_size, True)],
            dtype=object
        )
        self.__biases = array(
            [Layer(hidden_size, 1, True), Layer(output_size, 1, True)],
            dtype=object
        )

    def __sigmoid(self, value: int | float):
        # z = 1/(1+e^(-z));
        return 1.0 / (1.0 + exp(-value))

    def __dsigmoid(self, value: int | float):
        # z * (1 - z)
        return value * (1 - value)

    def train(self, data: array, epochs: int, eta: float):
        for epoch in range(0, epochs):
            random.shuffle(data)
            input_layer = Layer(1, self.__input_size)
            for record in range(0, len(data)):
                input_layer.load_array([data[record][0]])
                hidden_layer = self.__weights[0] * input_layer
                hidden_layer.sum_cols()
                hidden_layer += self.__biases[0]
                hidden_layer.load_list(
                    [self.__sigmoid(hidden_layer.matrix[row][0]) for row in range(0, hidden_layer.rows)]
                )

                output_layer = self.__weights[1] * hidden_layer.T
                output_layer.sum_cols()
                output_layer += self.__biases[1]
                output_layer.load_list(
                    [self.__sigmoid(output_layer.matrix[row][0]) for row in range(0, output_layer.rows)]
                )
