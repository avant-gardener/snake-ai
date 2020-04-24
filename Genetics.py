import NeuralNetwork
import random


def crossover(network_x, network_y):
    child_size = network_x.sizes
    child = NeuralNetwork.Network(child_size)
    # Bias crossover
    bias_length = len(child.biases)
    for i in range(bias_length):
        layer_biases = child.biases[i]
        layer_bias_length = len(layer_biases)
        for k in range(layer_bias_length):
            random_value = random.random()
            if random_value < 0.475:
                child.biases[i][k] = network_x.biases[i][k]
            elif random_value < 0.95:
                child.biases[i][k] = network_y.biases[i][k]
    # Weight crossover
    weight_length = len(child.weights)
    for i in range(weight_length):
        layer_nodes = child.weights[i]
        layer_length = len(layer_nodes)
        for k in range(layer_length):
            random_value = random.random()
            if random_value < 0.475:
                child.weights[i][k] = network_x.weights[i][k]
            elif random_value < 0.95:
                child.weights[i][k] = network_y.weights[i][k]
            """
            layer_weights = child.weights[i][k]
            layer_weight_length = len(layer_weights)
            for j in range(layer_weight_length):
                random_value = random.random()
                if random_value < 0.45:
                    child.weights[i][k][j] = network_x.weights[i][k][j]
                elif random_value < 0.9:
                    child.weights[i][k][j] = network_y.weights[i][k][j]
            """
    return child
