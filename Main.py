""" Main file for program """
import os
import numpy as np
import Population
import Settings


mode = input("Press 1 if you want to run the model \n "
             "Press 2 if you want to run past model's best snakes \n "
             "Press 3 if you want to run a specific snake from past model \n"
             "")


if mode == "1":
    generation_count = Settings.generation_count
    this_generation = Population.generate_random_population()
    print("First Generation Average Fitness: \
          {0}".format(Population.get_average(this_generation)))
    print(this_generation)
    for j in range(generation_count):
        this_generation = Population.new_generation(this_generation)
        print("Generation number: {0}  Average Fitness: {1}  "
              "Median Fitness: {2}"
              .format(j, Population.get_average(this_generation),
                      this_generation[(Settings.population_size + 1) // 2][1]))
        print(this_generation)
        np.save("Values/Weights/weight_output{0}"
                .format(j), this_generation[0][0].weights)
        np.save("Values/Biases/bias_output{0}"
                .format(j), this_generation[0][0].biases)
elif mode == "2":
    DIR = 'Values/Weights'
    generation_count = len([name for name in os.listdir(DIR)
                            if os.path.isfile(os.path.join(DIR, name))])
    for p in range(generation_count):
        w_data = np.load("Values/Weights/weight_output{0}.npy"
                         .format(p), allow_pickle=True)
        b_data = np.load("Values/Biases/bias_output{0}.npy"
                         .format(p), allow_pickle=True)
        print("Generation {0}".format(p))
        Population.load_snakes(w_data, b_data)
elif mode == "3":
    while 1:
        snake_selected = input()
        w_data = np.load("Values/Weights/weight_output{0}.npy"
                         .format(snake_selected), allow_pickle=True)
        b_data = np.load("Values/Biases/bias_output{0}.npy"
                         .format(snake_selected), allow_pickle=True)
        Population.load_snakes(w_data, b_data)
