import NeuralNetwork
import SnakeAI
import Genetics
import random
import numpy as np
import os


def generate_random_population():
    population_size = 100
    network_size = [6, 20, 3]
    generation = []
    for i in range(population_size):
        snake = NeuralNetwork.Network(network_size)
        fitness_of_snake = SnakeAI.game(True, snake)
        generation.append([snake, fitness_of_snake])
        generation.sort(key=lambda x: x[1], reverse=True)
    return generation


def new_generation(parents):
    population_size = len(parents)
    generation = []
    parents_for_new_generation = parents[:10]
    for i in range(population_size // 4):
        snake = parents[i][0]
        fitness_of_snake = SnakeAI.game(True, snake)
        generation.append([snake, fitness_of_snake])
        rand_value_x = random.randint(0, len(parents_for_new_generation) - 1)
        rand_value_y = random.randint(0, len(parents_for_new_generation) - 1)
        parent_x = parents_for_new_generation[rand_value_x][0]
        parent_y = parents_for_new_generation[rand_value_y][0]
        snake = Genetics.crossover(parent_x, parent_y)
        fitness_of_snake = SnakeAI.game(True, snake)
        generation.append([snake, fitness_of_snake])
    generation.sort(key=lambda x: x[1], reverse=True)
    return generation


def get_average(members):
    t_sum = 0
    for member in members:
        t_sum += member[1]
    return t_sum / len(members)


def load_snakes(weight_data, bias_data):
    snake = NeuralNetwork.Network([6, 20, 3])
    snake.biases = bias_data
    snake.weights = weight_data
    fitness_of_snake = SnakeAI.game(True, snake)
    return fitness_of_snake


mode = input("Press 1 if you want to run the model \n Press 2 if you want to run past models best \n "
             "Press 3 if you want to run a specific snake from past model")


if mode == "1":
    generation_number = 20
    this_generation = generate_random_population()
    print("First Generation Average Fitness: " + str(get_average(this_generation)))
    print(this_generation)
    for j in range(generation_number):
        this_generation = new_generation(this_generation)
        print("Generation number: " + str(j) + "   Generation Average Fitness: " + str(get_average(this_generation)))
        print(this_generation)
        np.save("Values/Weights/weight_output{0}".format(j), this_generation[0][0].weights)
        np.save("Values/Biases/bias_output{0}".format(j), this_generation[0][0].biases)
elif mode == "2":
    DIR = 'Values/Weights'
    generation_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    for p in range(generation_count):
        w_data = np.load("Values/Weights/weight_output{0}.npy".format(p), allow_pickle=True)
        b_data = np.load("Values/Biases/bias_output{0}.npy".format(p), allow_pickle=True)
        load_snakes(w_data, b_data)
elif mode == "3":
    while 1:
        snake_selected = input()
        w_data = np.load("Values/Weights/weight_output{0}.npy".format(snake_selected), allow_pickle=True)
        b_data = np.load("Values/Biases/bias_output{0}.npy".format(snake_selected), allow_pickle=True)
        load_snakes(w_data, b_data)
