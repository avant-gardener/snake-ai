import NeuralNetwork
import SnakeAI
import Genetics
import random
import Settings


def generate_random_population():
    population_size = Settings.population_size
    network_size = Settings.network_size
    generation = []
    for i in range(population_size):
        snake = NeuralNetwork.Network(network_size)
        fitness_of_snake = SnakeAI.game(True, snake, "fast")
        generation.append([snake, fitness_of_snake])
    generation.sort(key=lambda x: x[1], reverse=True)
    return generation


def new_generation(parents):
    population_size = len(parents)
    generation = []
    parents_for_new_generation = parents[:Settings.parent_count_to_new_generation]
    for i in range(population_size // 5):
        snake = parents[i][0]
        fitness_of_snake = SnakeAI.game(True, snake, "fast")
        generation.append([snake, fitness_of_snake])
        for b in range(4):
            rand_value_x = random.randint(0, len(parents_for_new_generation) - 1)
            rand_value_y = random.randint(0, len(parents_for_new_generation) - 1)
            parent_x = parents_for_new_generation[rand_value_x][0]
            parent_y = parents_for_new_generation[rand_value_y][0]
            snake = Genetics.crossover(parent_x, parent_y)
            fitness_of_snake = SnakeAI.game(True, snake, "fast")
            generation.append([snake, fitness_of_snake])
    generation.sort(key=lambda x: x[1], reverse=True)
    return generation


def get_average(members):
    t_sum = 0
    for member in members:
        t_sum += member[1]
    return t_sum / len(members)


def load_snakes(weight_data, bias_data):
    snake = NeuralNetwork.Network(Settings.network_size)
    snake.biases = bias_data
    snake.weights = weight_data
    fitness_of_snake = SnakeAI.game(True, snake, "middle")
    return fitness_of_snake
