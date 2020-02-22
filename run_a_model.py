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

comparator = MDStarComparator()
encoder = RealNumberEncoder()
mutator = CreepMutation(1, 0.1, encoder)
dataset = QSNMCDataset()
NeuronModel = IzencovichModel
population = Population(NeuronModel, 2, dataset)
population.initialize_population(encoder)
for individual in population.individuals:
    score = comparator.evaluate(individual, dataset.spike_trains)
    individual.fitness = score
for individual in population.individuals:
    mutator.mutate_individual(individual)
for individual in population.individuals:
    score = comparator.evaluate(individual, dataset.spike_trains)
    individual.fitness = score


# Initialize model
#iz_model = NeuronModel(a, b, c, d, dataset)
#encoder = RealNumberEncoder(NeuronModel.parameter_intervals)
#encoded_parameters = encoder.encode(iz_model)

#print(encoded_parameters)
#decoded_params = encoder.decode(iz_model, **encoded_parameters)

# comparator = MDStarComparator()
# iz_model.fitness = comparator.evaluate(iz_model, dataset.spike_trains)