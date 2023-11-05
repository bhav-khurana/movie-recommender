import numpy as np
import matplotlib.pyplot as plt
from utilities import *

PROBABILITY_OF_CROSSOVER = 0.9
PROBABILITY_OF_MUTATION = 0.01
POPULATION_SIZE = 200
TOURNAMENT_SIZE = 20
MAX_GENERATIONS = 1000
NUM_USERS = 943
NUM_MOVIES = 1682

rng = np.random.default_rng()

# Training data
ratings_base = np.genfromtxt('data/ua.base', dtype='int32')
matrix_base = np.zeros((NUM_USERS, NUM_MOVIES), dtype='int32')

# Testing data
ratings_test = np.genfromtxt('data/ua.test', dtype='int32')
matrix_test = np.zeros((NUM_USERS, NUM_MOVIES), dtype='int32')

# Creates a NUM_USERS X NUM_MOVIES matrix and assigns the rating provided by the dataset
# For all users that did not rate any movie, a random list of rating is added to the dataset
for r in ratings_base: 
    matrix_base[r[0]-1, r[1]-1] = r[2]
for r in ratings_test:
    matrix_test[r[0] - 1, r[1] - 1] = r[2]

matrix_filled = matrix_base.copy()
unrated = np.where(matrix_filled == 0)
matrix_filled[unrated] = np.random.randint(low=1, high=6, size=len(unrated[1]))

def fitness(x):
    return sum([pearsonr(x[neighborhood_rated[i]], neigh[neighborhood_rated[i]]) for i, neigh in enumerate(neighborhood)])

# Performs the Torunament Selection algorithm
# Np number of individuals are chosen, and then at each iteration, the best 2 individuals from the 
# randomly chosen individuals are selected to be the parents
def selection(pop_fitness):
    num_pairs = int(POPULATION_SIZE * PROBABILITY_OF_CROSSOVER / 2)     # Number of parent pairs that should be selected
    parents = np.empty((num_pairs, 2), dtype='int32')
    for idx in range(num_pairs):
        sample = rng.choice(POPULATION_SIZE, TOURNAMENT_SIZE, replace=False)
        max_pair = pop_fitness[sample].argsort()[-2:][::-1]
        parents[idx] = sample[max_pair]
    return parents

# Uniform Crossover with Crossover Mask
def crossover(parent_pairs):
    offspring = []
    for parent in parent_pairs:
        # Creation of the mask
        positions = np.random.randint(low=0, high=2, size=NUM_MOVIES)
        # Create the offsprings
        offspring.append(np.where(positions == 0, parent[0], parent[1]))
        offspring.append(np.where(positions == 0, parent[1], parent[0]))
    return offspring

# Random Mutation on Real Number GA Encoding
def mutation(chromosomes):
    num_mutations = int(PROBABILITY_OF_MUTATION * NUM_MOVIES)
    for chrom in chromosomes:
        positions = rng.choice(NUM_MOVIES, num_mutations, replace=False)
        values = np.random.randint(1, 6, num_mutations)
        chrom[positions] = values
        chrom[user_rated] = user[user_rated]

# Create a list of the users for which you want to analyse
random_users = [382]
for ind, current_user in enumerate(random_users):
    user = matrix_base[current_user]                                    # Ratings of all movies by the user
    user_rated = np.where(user != 0)                                    # Movies where were actually rated by the user
    
    user_test_pos = np.where(matrix_test[current_user] != 0)
    user_test = matrix_test[current_user][user_test_pos]                # Retireve the test ratings of the user

    user_unrated = np.where(user == 0)                                  # The movies which were not rated by user
    others = np.delete(matrix_base, current_user, axis=0)
    others_filled = np.delete(matrix_filled, current_user, axis=0)      # Other users with randomly filled values

    # Calculate the pair wise similarity with every user
    similarity = np.array([pearsonr(user[user_rated], us[user_rated]) for us in others_filled])

    # Selects the indices of the top 10 most similar users
    max_users = similarity.argsort()[-10:][::-1] 

    neighborhood = others[max_users]
    neighborhood_rated = [np.where(neigh != 0) for neigh in neighborhood]

    # Initialise empty arrays
    best = np.empty((MAX_GENERATIONS, 10))
    rmse = np.empty((MAX_GENERATIONS, 10))
    mae = np.empty((MAX_GENERATIONS, 10))
    
    optimals = []
    generations = []

    # Perform the algorithm 10 times
    for i in range(10):
        # Create initial population by inserting the user ratings for the rated movies
        initial_population = np.random.randint(low=1, high=6, size=(POPULATION_SIZE, NUM_MOVIES))
        for chromosome in initial_population:
            chromosome[user_rated] = user[user_rated]       

        next_pop = initial_population
        next_pop_fitness = np.array([fitness(x) for x in next_pop])
        next_sorted_keys = next_pop_fitness.argsort()
        
        generation = 0
        last_leader_change = 0
        earlier_performance = 0
        finished = False
        
        while not finished:
            current_pop = next_pop
            current_pop_fitness = next_pop_fitness
            current_sorted_keys = next_sorted_keys

            # Perform selection, crossover and mutation on children
            parents = selection(current_pop_fitness)
            children = crossover(current_pop[parents])
            mutation(children)

            # Store the modified population in the next population variables
            next_pop = current_pop.copy()
            next_pop[current_sorted_keys[:len(children)]] = np.array(children)
            next_pop_fitness = np.array([fitness(x) for x in next_pop])
            next_sorted_keys = next_pop_fitness.argsort()

            improvement = np.sum(next_pop_fitness) / np.sum(current_pop_fitness)
            best_sol = next_pop[next_sorted_keys[-1]]
            best_fitness = next_pop_fitness[next_sorted_keys[-1]]
            last_leader_change = generation if current_pop_fitness[current_sorted_keys[-1]] < best_fitness else last_leader_change

            # Store the metrics for the current generation
            best[generation][i] = best_fitness
            rmse[generation][i] = np.sqrt(mean_squared_error(best_sol[user_test_pos], user_test))
            mae[generation][i] = mean_absolute_error(best_sol[user_test_pos], user_test)

            # Terminate when MAX_GENRATIONS have passed or the `leader` (fittest indiviudal) has not changed since the last 50 generations
            finished = generation > MAX_GENERATIONS or generation - last_leader_change > 50
            # If there is less than 1% improvement since the last 100 generations, terminate
            if not finished and generation >= 100:
                finished = best[generation, i] / best[generation - 100, i] < 1.01
            generation += 1

        best[generation:, i] = max(next_pop_fitness)
        rmse[generation:, i] = rmse[generation-1, i]
        mae[generation:, i] = mae[generation-1, i]

        optimals.append(max(next_pop_fitness))
        generations.append(generation)

    best_avg = np.average(best[:max(generations)], axis=1)
    rmse_avg = np.average(rmse[:max(generations)], axis=1)
    mae_avg = np.average(mae[:max(generations)], axis=1)
    
    # Print the calculated results
    print('Average optimal:', np.average(np.array(optimals)))
    print('Average generations:', np.average(np.array(generations)))
    print('Average RMSE:', rmse_avg[-1])
    print('Average MAE:', mae_avg[-1])

    # Plot the charts for the metrics of the user across the number of generations
    plt.plot(np.arange(max(generations)), best_avg)
    plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
              + str(PROBABILITY_OF_MUTATION) + '%')
    plt.xlabel('Number of generations')
    plt.ylabel('Average fitness of best value')
    plt.grid(visible=True, axis='y')
    plt.show()

    plt.plot(np.arange(max(generations)), rmse_avg)
    plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
              + str(PROBABILITY_OF_MUTATION) + '%')
    plt.xlabel('Number of generations')
    plt.ylabel('Average RMSE of best value')
    plt.grid(visible=True, axis='y')
    plt.show()

    plt.plot(np.arange(max(generations)), mae_avg)
    plt.title('Population size: ' + str(POPULATION_SIZE) + ' Crossover: ' + str(PROBABILITY_OF_CROSSOVER) + '% Mutation: '
              + str(PROBABILITY_OF_MUTATION) + '%')
    plt.xlabel('Number of generations')
    plt.ylabel('Average MAE of best value')
    plt.grid(visible=True, axis='y')
    plt.show()
