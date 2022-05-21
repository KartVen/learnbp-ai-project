from numpy import array, random
from tabulate import tabulate
from layer import Layer
from matrix import Matrix


class Network(object):
    # Initialize a network
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        self.__layers = array([Layer(n_hidden, n_inputs), Layer(n_outputs, n_hidden)], dtype=object)

    def __forward_propagate(self, data_record: array):
        inputs = Matrix().load_list(data_record)[:]
        for layer in self.__layers:
            for neuron in range(layer.size):
                layer.activate(neuron, inputs)
                layer.sigmoid(neuron)
            inputs = layer.outputs
        return inputs

    def __dsigmoid(self, value: float):
        return value * (1.0 - value)

    def train(self, data: array, epochs: int, eta: float, test_data: array = None):
        for epoch in range(epochs):
            random.shuffle(data)
            for record in data:
                network_outputs = self.__forward_propagate(record[0])
                data_outputs = Matrix().load_list([record[1][i] for i in range(len(record[1]))])

                for layer in [1, 0]:
                    if layer == len(self.__layers) - 1:
                        for neuron in range(self.__layers[layer].size):
                            gradient = self.__dsigmoid(self.__layers[layer].outputs[neuron][0])
                            self.__layers[layer].error_delta[neuron][0] = (network_outputs[neuron][0] - data_outputs[neuron][0]) * gradient
                    else:
                        for neuron in range(self.__layers[layer].size):
                            error = float(0)
                            for neuron_forward in range(self.__layers[layer + 1].size):
                                error += (self.__layers[layer + 1].weights[neuron_forward][neuron] * self.__layers[layer + 1].error_delta[neuron_forward][0])
                            gradient = self.__dsigmoid(self.__layers[layer].outputs[neuron][0])
                            self.__layers[layer].error_delta[neuron][0] = error * gradient

                for layer in [1, 0]:
                    neuron_inputs = Matrix().load_list([record[0][i] for i in range(len(record[0]))])
                    if layer != 0:
                        neuron_inputs = self.__layers[layer - 1].outputs
                    for neuron in range(self.__layers[layer].size):
                        for branch in range(self.__layers[layer].inputs):
                            self.__layers[layer].weights[neuron][branch] -= self.__layers[layer].error_delta[neuron][0] * eta * neuron_inputs[branch][0]
                        self.__layers[layer].biases[neuron][0] -= self.__layers[layer].error_delta[neuron][0] * eta

            if test_data is not None:
                correct_predict, all_to_predict = self.count_predict(test_data), len(test_data)
                print(f"Learning - epoch: {epoch} | prediction: {correct_predict}/{all_to_predict}")
            else:
                print(f"Learning - epoch: {epoch}")

    def restore_output(self, value: float, min_value: float, max_value: float, y_min: float = 0, y_max: float = 1):
        return round(value * (max_value - min_value) / (y_max - y_min) + min_value)

    def count_predict(self, data: array):
        correct_predict = 0
        for record in data:
            # print(self.restore_output(self.__forward_propagate(record[0]).matrix[0][0], 1, 6), self.restore_output(record[1][0], 1, 6))
            print(self.__forward_propagate(record[0]).matrix[0][0], record[1][0])
            if self.restore_output(self.__forward_propagate(record[0]).matrix[0][0], 1, 6) == self.restore_output(record[1][0], 1, 6):
                correct_predict += 1
        return correct_predict

    def predict(self, data_input: array):
        return self.restore_output(self.__forward_propagate(data_input).matrix[0][0], 1, 6)
