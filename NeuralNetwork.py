import numpy as np
import random


def vectorized_result(p):
    e = np.zeros((10, 1))
    e[p] = 1.0
    return e


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(x):
    return max(0, x)


def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0


def cost_derivative(output_activations, y):
    return output_activations - y


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate, testing_data=None):
        n_test = 0
        if testing_data:
            n_test = len(testing_data)
        data_length = len(training_data)
        # Shuffle the data and divide it into mini batches, then update those batches with backprop
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0, data_length, mini_batch_size):
                mini_batches.append(training_data[k:k + mini_batch_size])
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if testing_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(testing_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, learning_rate):
        batch_bias = []
        batch_weight = []
        for bias in self.biases:
            batch_bias.append(np.zeros(bias.shape))
        for weight in self.weights:
            batch_weight.append(np.zeros(weight.shape))
        # Get new values with backprop
        # Update the batch bias with new values
        for x, y in mini_batch:
            delta_batch_bias, delta_batch_weight = self.backprop(x, y)
            counter = 0
            for bias, delta_bias in zip(batch_bias, delta_batch_bias):
                for k in range(len(delta_bias)):
                    batch_bias[counter][k] = bias[k] + delta_bias[k]
                counter += 1
            counter = 0
            for weight, delta_weight in zip(batch_weight, delta_batch_weight):
                for k in range(len(delta_weight)):
                    batch_weight[counter][k] = weight[k] + delta_weight[k]
                counter += 1
        # Update the real values of the model with batch values
        counter = 0
        for cur_bias, new_bias in zip(self.biases, batch_bias):
            for k in range(len(new_bias)):
                self.biases[counter][k] = cur_bias[k] - (learning_rate/len(mini_batch)) * new_bias[k]
            counter += 1
        counter = 0
        for cur_weight, new_weight in zip(self.weights, batch_weight):
            for k in range(len(new_weight)):
                self.weights[counter][k] = cur_weight[k] - (learning_rate/len(mini_batch)) * new_weight[k]
            counter += 1

    def backprop(self, x, y):
        temp_biases = []
        temp_weights = []
        for bias in self.biases:
            temp_biases.append(np.zeros(bias.shape))
        for weight in self.weights:
            temp_weights.append(np.zeros(weight.shape))
        # Feedforward
        activation = x
        activations = [x]
        z_layers = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            z_layers.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(z_layers[-1])
        temp_biases[-1] = delta
        temp_weights[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = z_layers[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            temp_biases[-layer] = delta
            temp_weights[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return temp_biases, temp_weights

    def evaluate(self, testing_data):
        test_results = []
        total = 0
        for (x, y) in testing_data:
            test_results.append((np.argmax(self.feedforward(x)), y))
        for (x, y) in test_results:
            if y[x] == 1.0:
                total += 1
        return total
