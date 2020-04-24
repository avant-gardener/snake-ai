import NeuralNetwork
import SnakeAI
import Genetics
import random


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
    for i in range(population_size // 2):
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


generation_number = 20
this_generation = generate_random_population()
print("First Generation Average Fitness: " + str(get_average(this_generation)))
print(this_generation)
for j in range(generation_number):
    this_generation = new_generation(this_generation)
    print("Generation number: " + str(j) + "   Generation Average Fitness: " + str(get_average(this_generation)))
    print(this_generation)
