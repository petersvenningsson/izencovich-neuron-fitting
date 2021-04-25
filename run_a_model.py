# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import time
import csv
import json
# 3rd party
import numpy as np
import matplotlib.pyplot as plt
# Local
from dataset.dataset_QSNMC import QSNMCDataset
from models.izencovich_neuron import SimpleIzencovichModel,IzencovichModelExtended
from models.LIF import LIFModel, LIF
from evaluators.MD_star import MDStarComparator
from GA.population import Population
from GA.encoders import RealNumberEncoder
from GA.mutators import CreepMutation
from GA.selection import TournamentCrossover
from GA.elitism import Elitism
import seaborn as sns
#############
# CONSTANTS #
#############
##########
# SCRIPT #
##########
comparator = MDStarComparator(delta = 1)
dataset = QSNMCDataset()
NeuronModel = IzencovichModelExtended

individual = NeuronModel(dataset,a=0.3017337265776834,b=0.14916560882651994,c=-59.33629988414421,d=106.57674262730798,v_rest=-54.61904283240068,v_threshold=-50.286389631670055,k=1.6259421609654119,C=125.52690852164866,polynomial_a=-0.07730008943907689, polynomial_b=-0.03043767818048064)

individual.simulate_spiking_times()
predicted_spike_times = individual.spike_times
spike_trains = dataset.spike_trains
score = comparator.evaluate(individual, spike_trains, voided_time = 16, voided_end = 32.7 ) # voided time validation 32.7 # voided time training 16, 32.7
print("score is {}".format(score))

def plot_input_output(v, I, title:str, filename:str):
    """ beautification of results.
    """
    sns.set(style = "dark")
    plt.rcParams.update({'font.size': 14})
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    time = [dataset.dt * x for x in range(0, len(I))]
    # Initialize
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(time, v, 'tab:blue', label='Membrane Potential', alpha = 0.4)
    ax1.set_xlabel('time (s)')
    #fig.suptitle(title, fontsize=8)

    # Plot output
    ax1.set_ylabel('Membrane Potential mV', color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')
    ax1.set_ylim(-150, 55)
    ax2 = ax1.twinx()

    # Plot input current
    ax2.plot(time, I, 'tab:red', label='Input Current', alpha = 0.4)
    ax2.set_ylim(-25, 25)
    ax2.set_xlim(25,28)
    ax1.set_xlim(25, 28)
    ax2.set_ylabel('Input Current (pA)', color='tab:red')
    ax2.tick_params('y', colors='tab:red')

    fig.tight_layout()
    ax1.legend(loc=1)
    ax2.legend(loc=3)
    plt.savefig(filename+'.jpg')


def plot_voltage_voltage(individual, dataset, data_sample: int, title:str, filename:str, limits = (30,36)):
    """
    """
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    time = [dataset.dt * x for x in range(0, len(individual.i_ext))]
    predicted_voltage = individual.v
    observed_voltage = dataset.gold_voltages[data_sample]
    sns.set(style = "dark")
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    # Initialize
    fig, ax1 = plt.subplots(figsize=(16, 6))
    fig.subplots_adjust(right=0.75)
    fig.subplots_adjust(bottom=0.2)
    ax1.plot(time, np.zeros_like(time), 'w', alpha = 0.8)
    ax1.plot(time, np.ones_like(time)*50, 'w', alpha=0.8)
    ax1.plot(time, np.ones_like(time)*(-50), 'w', alpha=0.8)
    ax1.plot(time, np.ones_like(time)*(100), 'w', alpha=0.8)
    ax1.plot(time, np.ones_like(time)*(-100), 'w', alpha=0.8)

    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    ax1.plot(time, predicted_voltage, 'tab:blue', label='Predicted voltage', alpha = 0.5)
    ax1.plot(individual.spike_times, np.ones_like(individual.spike_times)*60, 'bo', label='Spiking times for predicted voltage', alpha=0.5)
    plt.savefig('individual.jpg')

    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    ax1.set_xlabel('time (s)',  fontsize=22)

    # Plot output
    ax1.set_ylabel('Predicted voltage (mV)',  fontsize=22, color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')
    ax1.set_ylim(-140, 120)
    ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    # Plot observed voltage
    ax2.plot(time, observed_voltage, 'tab:green', label='Observed voltage', alpha = 0.5)
    ax2.plot(dataset.spike_trains[data_sample], np.ones_like(dataset.spike_trains[data_sample])*45, 'go', label='Spiking times for observed voltage', alpha=0.5)
    ax2.set_ylim(-140, 120)
    ax2.set_ylabel('Observed voltage (mV)', fontsize=22, color='tab:green')
    ax2.tick_params('y', colors='tab:green')
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    ax1.legend(loc=1, prop={'size': 14})
    ax2.legend(loc=3, prop={'size': 14})
    ax1.set_xlim(*limits)

    plt.savefig(filename+'.jpg')

# Plot measured data
plot_input_output(dataset.gold_voltages[-1], dataset.current, "", 'measured_voltage')

#Plot model performance
plot_input_output(
    individual.v, individual.i_ext, "The score achieved is " + str(round(score,5)), 'summary_of_model_results'
)
# Plot comparison
dataset_sample = 12
for limits in [(12.5,16), (18,24), (24,30), (32.7,36), (36,39)]:
    title = "Sample {} - Comparison of observed to predicted voltage with score {} \n" \
            " {} predicted spikes, {:0.1f} average observed spikes".format(
        dataset_sample,str(round(score,4)), len(individual.spike_times), np.mean([len(train) for train in dataset.spike_trains]))

    filename = "predicted_observed_comparison sample {}, {}".format(dataset_sample,str(limits))
    plot_voltage_voltage(individual, dataset, dataset_sample, title, filename, limits)




