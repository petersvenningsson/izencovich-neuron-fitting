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
from models.izencovich_neuron import SimpleIzencovichModel,IzencovichModelExtended
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

#spikes_at_all_times = np.asarray([x*(1e-4) for x in range(int(39/1e-4))])
# High fitness individual
#individual = SimpleIzencovichModel(dataset, 0.017657556305599773,0.21249900146986894,-55.34256372605247,5.388732724289586)
individual = NeuronModel(dataset, a = 0.05192601802452859,b = 3.164746139631182,c = -61.405388566159715,d = 91.3996081711308,C = 53.05862361800592,v_rest = -52.689041287250554,v_threshold = -43.62016618987195, k = 0.6253421048255764)
#individual = NeuronModel(dataset, a = 0.026567448540741473,b= -1.918465458596826,c = -49.614407461361324, d= 93.46694934559687,C = 101.46238923693426,v_rest = -60.064935754463505,v_threshold=-49.61239092069363,k=0.5793442056713082)
individual.simulate_spiking_times()
predicted_spike_times = individual.spike_times

score = comparator.evaluate(individual, [dataset.spike_trains[0]])


def plot_input_output(v, I, title:str, filename:str):
    """ beautification of results.
    """
    time = [dataset.dt * x for x in range(0, len(I))]
    # Initialize
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(time, v, 'tab:blue', label='Output', alpha = 0.4)
    ax1.set_xlabel('time (ms)')
    fig.suptitle(title, fontsize=8)

    # Plot output
    ax1.set_ylabel('Output mV', color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')
    ax1.set_ylim(-150, 55)
    ax2 = ax1.twinx()

    # Plot input current
    ax2.plot(time, I, 'tab:red', label='Input', alpha = 0.4)
    ax2.set_ylim(-25, 25)
    ax2.set_ylabel('Input (pA)', color='tab:red')
    ax2.tick_params('y', colors='tab:red')

    fig.tight_layout()
    ax1.legend(loc=1)
    ax2.legend(loc=3)
    plt.savefig(filename+'.jpg')

def plot_voltage_voltage(individual, dataset, data_sample: int, title:str, filename:str, limits = (30,36)):
    """
    """
    time = [dataset.dt * x for x in range(0, len(individual.i_ext))]
    predicted_voltage = individual.v
    observed_voltage = dataset.gold_voltages[data_sample]

    sns.set(style = "dark")

    # Initialize
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(time, np.zeros_like(time), 'w', alpha = 0.8)
    ax1.plot(time, np.ones_like(time)*50, 'w', alpha=0.8)
    ax1.plot(time, np.ones_like(time)*(-50), 'w', alpha=0.8)
    ax1.plot(time, np.ones_like(time)*(100), 'w', alpha=0.8)
    ax1.plot(time, np.ones_like(time)*(-100), 'w', alpha=0.8)

    ax1.plot(time, predicted_voltage, 'tab:blue', label='predicted voltage', alpha = 0.5)
    ax1.plot(individual.spike_times, np.ones_like(individual.spike_times)*60, 'bo', label='Spiking times for predicted voltage', alpha=0.5)
    plt.savefig('individual.jpg')

    ax1.set_xlabel('time (ms)')
    fig.suptitle(title, fontsize=12)

    # Plot output
    ax1.set_ylabel('predicted voltage (mV)', color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')
    ax1.set_ylim(-140, 120)
    ax2 = ax1.twinx()

    # Plot input current
    ax2.plot(time, observed_voltage, 'tab:green', label='observed voltage', alpha = 0.5)
    ax2.plot(dataset.spike_trains[data_sample], np.ones_like(dataset.spike_trains[data_sample])*45, 'go', label='Spiking times for observed voltage', alpha=0.5)
    ax2.set_ylim(-140, 120)
    ax2.set_ylabel('observed voltage (mV)', color='tab:green')
    ax2.tick_params('y', colors='tab:green')

    #fig.tight_layout()
    ax1.legend(loc=1)
    ax2.legend(loc=3)

    #ax1.set_xlim(30, 36)
    ax1.set_xlim(*limits)

    plt.savefig(filename+'.jpg')

# Plot measured data
plot_input_output(dataset.gold_voltages[-1], dataset.current, "measured data", 'measured_voltage')

#Plot model performance

plot_input_output(
    individual.v, individual.i_ext, "The score achieved is " + str(round(score,5)), 'summary_of_model_results'
)
# Plot comparison
dataset_sample = 12
for limits in [(12.5,16), (18,24), (24,30), (30,36)]:
    title = "Sample {} - Comparison of observed to predicted voltage with score {} \n" \
            " {} predicted spikes, {:0.1f} average observed spikes".format(
        dataset_sample,str(round(score,4)), len(individual.spike_times), np.mean([len(train) for train in dataset.spike_trains]))

    filename = "predicted_observed_comparison sample {}, {}".format(dataset_sample,str(limits))
    plot_voltage_voltage(individual, dataset, dataset_sample, title, filename, limits)




