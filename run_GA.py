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
import argparse
# Local
from dataset.dataset_QSNMC import QSNMCDataset
from models.izencovich_neuron import IzencovichModelExtended, SimpleIzencovichModel
from models.LIF import LIF
from evaluators.MD_star import MDStarComparator
from GA.population import Population
from GA.encoders import RealNumberEncoder
from GA.mutators import CreepMutation
from GA.selection import TournamentCrossover
from GA.elitism import Elitism
#############
# CONSTANTS #
#############

voided_time=16
voided_end = 32.7
n_generations = 150
population_size=30

models = ['SimpleIzencovichModel', 'IzencovichModelExtended', 'LIF']
#############
# ARGPARSER #
#############
my_parser = argparse.ArgumentParser(description='Define the model to be fit')

my_parser.add_argument('--model',
                       type=str,
                       default='IzencovichModelExtended',
                       help='Please select one of: ' + ', '.join(models),
                       nargs='+')

my_parser.add_argument('--seed',
                       type=bool,
                       default=False,
                       help='Indicates if the population will be seeded from a pre-defined set.' +\
                       'Please select one of: True, False',
                       nargs='+')
args = my_parser.parse_args()

##########
# SCRIPT #
##########
selector = TournamentCrossover()
comparator = MDStarComparator(delta = 1)
encoder = RealNumberEncoder()
elitism = Elitism()
mutator = CreepMutation(encoder, mutation_probability=0.14, gaussian_creep=True, creep_rate = 0.1)
dataset = QSNMCDataset()
NeuronModel = globals[args.model]
population = Population(NeuronModel,  population_size, dataset)

if args.seed:
    population.initialize_population_seeded(encoder)
else:
    population.initialize_population(encoder)

max_fitness_history = []
mean_fitness_history = []

for i_generation in range(n_generations):
    print("Current generation is {}.".format(i_generation))
    try:
        print("The most fit individual has score {}".format(
            population.most_fit_individual.fitness
        ))
    except AttributeError:
        pass

    for i_individual, individual in enumerate(population.individuals):
        score = comparator.evaluate(individual, dataset.spike_trains,  voided_time = voided_time, voided_end=voided_end)
        individual.fitness = score
        print("Individual {} has score {}".format(i_individual, individual.fitness
        ))
        # If the score is lesser than 1% then the solution is likely divergent and can be discarded.
        while score < 0.01:
            if args.seed:
                new_individual = population.initialize_individual_seeded(encoder)
            else:
                new_individual = population.initialize_individual(encoder)
            score = comparator.evaluate(new_individual, dataset.spike_trains, voided_time = voided_time, voided_end=voided_end)
            new_individual.fitness = score
            population.individuals[i_individual] = new_individual
            assert new_individual in population.individuals
            print(
                "Individual {} has score {}".format(i_individual, new_individual.fitness
            ))
    population.set_most_fit_individual()

    max_fitness_history.append(population.most_fit_individual.fitness)
    with open('best_chromosome.txt', 'w') as file:
        wr = csv.DictWriter(file, fieldnames=population.most_fit_individual.__dict__.keys())
        wr.writerow(population.most_fit_individual.__dict__)
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

