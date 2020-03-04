# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import csv
import json
# 3rd party
import numpy as np
import matplotlib.pyplot as plt
# Local
from dataset.dataset_QSNMC import QSNMCDataset
from models.izencovich_neuron import IzencovichModelExtended
from evaluators.MD_star import MDStarComparator
from GA.population import Population
from GA.encoders import RealNumberEncoder
from GA.mutators import CreepMutation
from GA.selection import TournamentCrossover
from GA.elitism import Elitism
import seaborn as sns

class tempdata:
    def __init__(self):
        self.dt = 1e-4
        self.time = np.arange(0, 1, 1e-4)
        self.gold_voltages = None
        self.current = None
        self.initialize_current()

    def initialize_current(self):
        input_onset = 0.1
        I = np.zeros((len(self.time)))  # CURRENT (INPUT)
        for k in range(0, len(self.time)):
            if self.time[k] > input_onset:
                I[k] = 70  # Input change
        self.current = I

dataset = tempdata()
individual = IzencovichModelExtended(dataset,a=0.03, b=-2, c=-50, d=100, v_rest = -60, v_threshold = -40, k = 0.7, C = 100)
individual.simulate_spiking_times()
print("he")

def plot_input_output(v, I, title:str, filename:str):
    """ beautification of results.
    """
    time = [dataset.dt * x for x in range(0, len(I))]
    # Initialize
    fig, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(time, v, 'tab:blue', label='Output', alpha = 0.4)
    ax1.set_xlabel('time (ms)')
    fig.suptitle(title, fontsize=8)

    # Plot output
    ax1.set_ylabel('Output mV', color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')
    #ax1.set_ylim(-150, 55)
    ax2 = ax1.twinx()

    # Plot input current
    ax2.plot(time, I, 'tab:red', label='Input', alpha = 0.4)
    #ax2.set_ylim(-2500, 2500)
    ax2.set_ylabel('Input (mA)', color='tab:red')
    ax2.tick_params('y', colors='tab:red')

    fig.tight_layout()
    ax1.legend(loc=1)
    ax2.legend(loc=3)
    plt.savefig(filename+'.jpg')

plot_input_output(individual.v, individual.i_ext, "Recreating izhikevich", "recreation")