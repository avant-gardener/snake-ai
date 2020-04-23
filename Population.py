import NeuralNetwork
import SnakeAI
import Genetics
import random


def generate_random_population():
    population_size = 5
    network_size = [11, 10, 3]
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
    parents_for_new_generation = parents[:3]
    for i in range(population_size):
        rand_value_x = random.randint(0, len(parents_for_new_generation) - 1)
        rand_value_y = random.randint(0, len(parents_for_new_generation) - 1)
        parent_x = parents_for_new_generation[rand_value_x][0]
        parent_y = parents_for_new_generation[rand_value_y][0]
        snake = Genetics.crossover(parent_x, parent_y)
        fitness_of_snake = SnakeAI.game(True, snake)
        generation.append([snake, fitness_of_snake])
        generation.sort(key=lambda x: x[1], reverse=True)
    return generation


my_generation = generate_random_population()
print(my_generation)
generation_2 = new_generation(my_generation)
print(generation_2)
