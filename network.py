from numpy import random, exp, dot, zeros, argmax, array, linalg


class Network(object):

    # Constructor, takes list of layers with amount of neurons
    def __init__(self, layer_sizes):
        random.seed(1)
        self.layer_sizes = layer_sizes

        # Generating random value for weight and biases
        self.biases = [random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def predict(self, inputs):
        # Predict outputs of the network / feedforward
        for b_value, w_value in zip(self.biases, self.weights):
            inputs = self.__sigmoid(dot(w_value, inputs) + b_value)
        return inputs

    def train(self, training_data, mini_batch_size, epochs, eta, error_target, test_data=None):
        for i_epoch in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[n:n + mini_batch_size] for n in range(0, training_data.__len__(), mini_batch_size)
            ]

            for mini_batch in mini_batches:
                biases_gradient = [zeros(b.shape) for b in self.biases]
                weights_gradient = [zeros(w.shape) for w in self.weights]

                for i_record in range(mini_batch.__len__()):
                    # Counting the gradient for the cost function C_x.
                    biases_gradient_delta = [zeros(b_value.shape) for b_value in self.biases]
                    weights_gradient_delta = [zeros(w_value.shape) for w_value in self.weights]

                    # Feedforward Pass
                    # List all outputs of layers
                    outputs = [mini_batch[i_record][0]]
                    # List all excitations of layers
                    excitations = []

                    # Calculate activations for neuron
                    for w_value, b_value in zip(self.weights, self.biases):
                        excitation = dot(w_value, outputs[outputs.__len__() - 1])
                        excitation = excitation + b_value
                        excitations.append(excitation)
                        outputs.append(self.__sigmoid(excitation))

                    # Backward pass
                    error_output_layer = self.__cost_derivative(outputs[outputs.__len__() - 1], mini_batch[i_record][1]) * self.__d_sigmoid(excitations[excitations.__len__() - 1])
                    biases_gradient_delta[biases_gradient_delta.__len__() - 1] = error_output_layer
                    weights_gradient_delta[weights_gradient_delta.__len__() - 1] = dot(error_output_layer, outputs[outputs.__len__() - 2].transpose())

                    # Specifying the gradient increase for input and hidden layers
                    for i_layer in range(2, self.layer_sizes.__len__()):
                        excitation = excitations[excitations.__len__() - i_layer]
                        excitation_derivative = self.__d_sigmoid(excitation)
                        error_output_layer = dot(self.weights[self.weights.__len__() - i_layer + 1].transpose(), error_output_layer) * excitation_derivative
                        biases_gradient_delta[biases_gradient_delta.__len__() - i_layer] = error_output_layer
                        weights_gradient_delta[weights_gradient_delta.__len__() - i_layer] = dot(error_output_layer, outputs[outputs.__len__() - i_layer - 1].transpose())

                    biases_gradient = [
                        b_g_value + b_g_d_value for b_g_value, b_g_d_value in zip(biases_gradient, biases_gradient_delta)
                    ]
                    weights_gradient = [
                        w_g_value + w_g_d_value for w_g_value, w_g_d_value in zip(weights_gradient, weights_gradient_delta)
                    ]

                # Update weights
                self.weights = [
                    w_value - w_g_value * (eta / 2) for w_value, w_g_value in zip(self.weights, weights_gradient)
                ]

                # Update biases
                self.biases = [
                    b_value - b_g_value * (eta / 2) for b_value, b_g_value in zip(self.biases, biases_gradient)
                ]

            if test_data:
                error_current = round(0.5 * sum([pow(linalg.norm(self.predict(x) - y), 2) for (x, y) in test_data]), 6)
                predict_correctly = self.__check_accuracy(test_data)
                predict_correctly_acc = round(predict_correctly[0] / predict_correctly[1] * 100, 2)
                print("Epoch %s | %s | %s / %s | %s%% | N: %s %s | Error %s | mb: %s" % (
                    i_epoch, eta, predict_correctly[0], predict_correctly[1], predict_correctly_acc, self.layer_sizes[1], self.layer_sizes[2], error_current, mini_batch_size))

                if error_current < error_target or i_epoch == epochs - 1:
                    return None

            else:
                print("Epoch %s ..." % i_epoch)

    def __check_accuracy(self, test_data):
        network_results = [(self.predict(x), y) for (x, y) in test_data]
        predict_correctly = 0
        for i_record_test in range(network_results.__len__()):
            if argmax(network_results[i_record_test][0].T) == argmax(array(test_data[i_record_test][1])).T:
                predict_correctly += 1
        return predict_correctly, network_results.__len__()

    def __cost_derivative(self, output_activations, y):
        # Return error between the neuron and the real result
        return output_activations - y

    def __sigmoid(self, value):
        # The sigmoid function
        return 1.0 / (1.0 + exp(-value))

    def __d_sigmoid(self, value):
        # Derivative of the sigmoid function
        return self.__sigmoid(value) * (1 - self.__sigmoid(value))
