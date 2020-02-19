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
from dataset.dataset_QSNMC import QSNMC
from models.izencovich_neuron import IzencovichModel
from evaluators.MD_star import MDStarComparator
from geneutility.population import Population

#############
# CONSTANTS #
#############
a = 0.02
b = 0.2
c = -50
d = 2
v_spike = 30
##########
# SCRIPT #
##########

# Load dataset
dataset = QSNMC()
#population = Population(IzencovichModel, 10, dataset)
#population.initialize_population()
#print("hej")
# Initialize model
iz_model = IzencovichModel(a, b, c, d, dataset)
print("a")
# iz_model.set_parameters(a=0.03)

# comparator = MDStarComparator()
# iz_model.fitness = comparator.evaluate(iz_model, dataset.spike_trains)
