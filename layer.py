from numpy import exp
from matrix import Matrix


class Layer(object):
    def __init__(self, n_size, n_inputs):
        self.size = n_size
        self.inputs = n_inputs
        self.weights = Matrix(n_size, n_inputs, True)
        self.biases = Matrix(n_size, 1, True)
        self.outputs = Matrix(n_size, 1)
        self.error_delta = Matrix(n_size, 1)

    def activate(self, neuron: int, inputs: any):
        self.outputs[neuron][0] = self.biases[neuron]
        for i in range(self.inputs - 1):
            self.outputs[neuron][0] += self.weights[neuron][i] * inputs[i][0]

    def sigmoid(self, neuron: int, self_modify: bool = True, out: bool = False):
        if self_modify:
            self.outputs[neuron][0] = 1.0 / (1.0 + exp(-self.outputs[neuron]))
            if out:
                return self.outputs[neuron][0]

    def __sizeof__(self):
        return self.size
