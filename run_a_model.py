###########
# IMPORTS #
###########
# 3rd party
import matplotlib.pyplot as plt
import numpy as np
# Local
from dataset.dataset_QSNMC import QSNMC
from models.izencovich_neuron import IzencovichModel
from evaluators.MD_star import MDStarComparator


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

# Initialize model
iz_model = IzencovichModel(a, b, c, d, v_spike)
# Generate spike train
predicted_spike_time_train = iz_model.get_spike_train(dataset)
observed_spike_time_trains = dataset.spike_trains

comparator = MDStarComparator()
score = comparator.evaluate(observed_spike_time_trains, predicted_spike_time_train)
