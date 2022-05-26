from numpy import random, exp, dot, zeros, argmax


class Network(object):

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.biases = [random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def predict(self, inputs):
        # Predict outputs of the network
        for i in range(self.biases.__len__()):
            inputs = self.__sigmoid(dot(self.weights[i], inputs) + self.biases[i])
        return inputs

    def train(self, training_data, mini_batch_size, epochs, eta, test_data=None):
        for i_epoch in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[n:n + mini_batch_size] for n in range(0, training_data.__len__(), mini_batch_size)
            ]

            for mini_batch in mini_batches:
                biasesGradient = [zeros(b.shape) for b in self.biases]
                weightsGradient = [zeros(w.shape) for w in self.weights]

                for i_record in range(mini_batch.__len__()):
                    # Counting the gradient for the cost function C_x.
                    biasesGradientDelta = [zeros(b.shape) for b in self.biases]
                    weightsGradientDelta = [zeros(w.shape) for w in self.weights]

                    # Feedforward Pass
                    # List all outputs of layers
                    outputs = [mini_batch[i_record][0]]
                    # List all weightedSums of layers
                    weightedSums = []
                    for i_layer in range(self.biases.__len__()):
                        weightedSum = dot(self.weights[i_layer], outputs[outputs.__len__() - 1])
                        weightedSum = weightedSum + self.biases[i_layer]
                        weightedSums.append(weightedSum)
                        outputs.append(self.__sigmoid(weightedSum))

                    # Backward pass
                    errorOutputLayer = self.__cost_derivative(outputs[outputs.__len__() - 1], mini_batch[i_record][1]) * self.__d_sigmoid(weightedSums[weightedSums.__len__() - 1])
                    biasesGradientDelta[biasesGradientDelta.__len__() - 1] = errorOutputLayer
                    weightsGradientDelta[weightsGradientDelta.__len__() - 1] = dot(errorOutputLayer, outputs[outputs.__len__() - 2].transpose())

                    for i_layer in range(2, self.layer_sizes.__len__()):
                        weightedSum = weightedSums[weightedSums.__len__() - i_layer]
                        weightedSumDerivative = self.__d_sigmoid(weightedSum)
                        errorOutputLayer = dot(self.weights[self.weights.__len__() - i_layer + 1].transpose(), errorOutputLayer) * weightedSumDerivative
                        biasesGradientDelta[biasesGradientDelta.__len__() - i_layer] = errorOutputLayer
                        weightsGradientDelta[weightsGradientDelta.__len__() - i_layer] = dot(errorOutputLayer, outputs[outputs.__len__() - i_layer - 1].transpose())

                    for i_layer in range(biasesGradient.__len__()):
                        biasesGradient[i_layer] = biasesGradient[i_layer] + biasesGradientDelta[i_layer]
                    for i_layer in range(weightsGradient.__len__()):
                        weightsGradient[i_layer] = weightsGradient[i_layer] + weightsGradientDelta[i_layer]

                # Update weights
                for i_layer in range(self.weights.__len__()):
                    self.weights[i_layer] = self.weights[i_layer] - weightsGradient[i_layer] * (eta / len(mini_batch))

                # Update biases
                for i_layer in range(self.biases.__len__()):
                    self.biases[i_layer] = self.biases[i_layer] - biasesGradient[i_layer] * (eta / len(mini_batch))

            if test_data:
                predict_correctly = self.__check_efficiency(test_data)
                print("Epoch %s | %s | %s / %s | %f %%" % (i_epoch, eta, predict_correctly[0], predict_correctly[1], (predict_correctly[0] / predict_correctly[1] * 100)))
            else:
                print("Epoch %s ..." % i_epoch)

    def __check_efficiency(self, test_data):
        network_results = [(self.predict(x), y)
                           for (x, y) in test_data]
        predict_correctly = 0
        for i_record_test in range(network_results.__len__()):
            if argmax(network_results[i_record_test][0].T) == test_data[i_record_test][1][0][0]:
                predict_correctly += 1
        return predict_correctly, network_results.__len__()

    def __cost_derivative(self, output_activations, y):
        # Vector of partial derivatives dC_x/da for the output activations
        return output_activations - y

    def __sigmoid(self, value):
        # The sigmoid function
        return 1.0 / (1.0 + exp(-value))

    def __d_sigmoid(self, value):
        # Derivative of the sigmoid function
        return self.__sigmoid(value) * (1 - self.__sigmoid(value))
