# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import json
# 3rd party
import matplotlib.pyplot as plt
import numpy as np
# Local
from dataset.dataset_QSNMC import QSNMCDataset
from models.izencovich_neuron import IzencovichModel
from evaluators.MD_star import MDStarComparator
from GA.population import Population
from GA.encoders import RealNumberEncoder
from GA.mutators import CreepMutation
from GA.selection import TournamentCrossover
from GA.elitism import Elitism
#############
# CONSTANTS #
#############
a = 0.02
b = 0.2
c = -50
d = 2
v_max = 30
##########
# SCRIPT #
##########
selector = TournamentCrossover()
comparator = MDStarComparator()
encoder = RealNumberEncoder()
elitism = Elitism()
mutator = CreepMutation(encoder)
dataset = QSNMCDataset()
NeuronModel = IzencovichModel
population = Population(NeuronModel, population_size = 2, dataset)
population.initialize_population(encoder)

for individual in population.individuals:
    score = comparator.evaluate(individual, dataset.spike_trains)
    individual.fitness = score

population.set_most_fit_individual()
next_generation = selector.population_crossover(population)

population.individuals = next_generation
for individual in population.individuals:
    mutator.mutate_individual(individual)
elitism.elitism(population)

for individual in population.individuals:
    score = comparator.evaluate(individual, dataset.spike_trains)
    individual.fitness = score


