# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import copy
import csv
import json
# 3rd party
import numpy as np
import matplotlib.pyplot as plt
# Local
from dataset.dataset_QSNMC import QSNMCDataset
from models.izencovich_neuron import IzencovichModelExtended, SimpleIzencovichModel
from evaluators.MD_star import MDStarComparator
from GA.population import Population
from GA.encoders import RealNumberEncoder
from GA.mutators import CreepMutation
from GA.selection import TournamentCrossover
from GA.elitism import Elitism
#############
# CONSTANTS #
#############
n_generations = 60
population_size = 30
# SCRIPT #
##########
selector = TournamentCrossover()
comparator = MDStarComparator(delta = 1)
encoder = RealNumberEncoder()
elitism = Elitism()
mutator = CreepMutation(encoder, mutation_probability=0.14, gaussian_creep=True, creep_rate = 0.01)
dataset = QSNMCDataset()
NeuronModel = IzencovichModelExtended
population = Population(NeuronModel,  population_size, dataset)
population.initialize_population(encoder)

dataset_spike_train = [dataset.spike_trains[0]]
max_fitness_history = []
mean_fitness_history = []

for i_generation in range(n_generations):
    print("Current generation is {}.".format(i_generation))
    try:
        print("The most fit individual has parameters a: {:f}, b: {:f}, c: {:f}, d: {:f} with id: {} and score {}".format(
            population.most_fit_individual.a, population.most_fit_individual.b, population.most_fit_individual.c,
            population.most_fit_individual.d, id(population.most_fit_individual), population.most_fit_individual.fitness
        ))
    except AttributeError:
        pass

    for i_individual, individual in enumerate(population.individuals):
        score = comparator.evaluate(individual, dataset_spike_train)
        individual.fitness = score
        print("Individual {} has parameters a: {:f}, b: {:f}, c: {:f}, d: {:f} with id: {} and score {}".format(
            i_individual, individual.a, individual.b, individual.c, individual.d, id(individual), individual.fitness
            )
        )
        # If the score is lesser than 1% then the solution is likely divergent and can be discarded.
        while score < 0.01:
            new_individual = population.initialize_individual(encoder)
            score = comparator.evaluate(new_individual, dataset_spike_train)
            new_individual.fitness = score
            population.individuals[i_individual] = new_individual
            assert new_individual in population.individuals
            print("Individual {} has parameters a: {:f}, b: {:f}, c: {:f}, d: {:f} with id: {} and score {}".format(
                i_individual, new_individual.a, new_individual.b, new_individual.c, new_individual.d, id(new_individual), new_individual.fitness
            ))
    population.set_most_fit_individual()

    max_fitness_history.append(population.most_fit_individual.fitness)
    mean_fitness_history.append(np.mean([individual.fitness for individual in population.individuals]))

    next_generation = selector.population_crossover(population)

    for individual in next_generation:
        mutator.mutate_individual(individual)
    population.individuals = next_generation
    elitism.elitism(population)


with open('max_fitness.txt', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(max_fitness_history)

with open('mean_fitness.txt', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(mean_fitness_history)

with open('best_chromosome.txt', 'w') as file:
    wr = csv.DictWriter(file, fieldnames = population.most_fit_individual.__dict__.keys())
    wr.writerow(population.most_fit_individual.__dict__)

