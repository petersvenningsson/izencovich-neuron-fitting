###########
# IMPORTS #
###########
# 3rd party
import matplotlib.pyplot as plt
import numpy as np
# Local
from dataset.dataset_QSNMC import QSNMC
from models.izencovich_neuron import IzencovichModel
from scipy.signal import fftconvolve


#############
# CONSTANTS #
#############

##########
# SCRIPT #
##########

# Load dataset
dataset = QSNMC()

# Initialize model
a = 0.02
b = 0.2
c = -50
d = 2
v_spike = 30
iz_model = IZModel(a, b, c, d, v_spike, name='Izhikevich')

# Generate spike train
simulated_spike_train = iz_model.get_spike_train(dataset.current)


TMAX = 39
predicted = simulated_spike_train
observed = dataset.spike_trains

observed_spike_time_trains = [x.spike_times for x in observed]
predicted_spike_time_train = predicted.spike_times



T = TMAX
delta=1
dt = dataset.dt

# TODO: Create evaluator class and move this stuff there. Try to find good description of MD.
# TODO: Seems like AnalogSignal class is not needed. Try to decouple
# TODO: Seems like Neuron class is not needed. Try to decouple.
# TODO: Write stronger doc strings.

def Md_dot_product(spike_train_1, spike_train_2, delta, dt):
    rectangular_size = 2*int(delta/dt)
    rectangular_window = np.ones(rectangular_size)
    spike_train_1_filtered = fftconvolve(spike_train_1, rectangular_window, mode='same')
    spike_train_2_filtered = fftconvolve(spike_train_2, rectangular_window, mode='same')

    dot_product_value = np.dot(spike_train_1_filtered, spike_train_2_filtered)
    return dot_product_value


def computeMD(delta, dt):
    """ MD* proposed by Richard Naud as a spike train similarity measure.
     See Improved Similarity Measures for Small Sets of Spike Trains, Richard Naud et al. for
     more information."""
    print("Computing Md* {} ms precision...".format(delta))
    observed_spike_trains = []
    
    n_trains = len(observed_spike_time_trains)

    for spike_times in observed_spike_time_trains:
        spike_train = getSpikeTrain(spike_times, TMAX, dt)
        observed_spike_trains.append(spike_train)

    observed_average_spike_train = getAverageSpikeTrain(observed_spike_time_trains, TMAX, dt)
    predicted_spike_train = getSpikeTrain(predicted_spike_time_train, TMAX, dt)

    # Compute dot product <data, model>
    dot_product_observed_predicted = Md_dot_product(observed_average_spike_train, predicted_spike_train, delta, dt=dt)

    # Compute dot product <model, model>
    dot_product_predicted_predicted = Md_dot_product(predicted_spike_train, predicted_spike_train, delta, dt=dt)

    # Compute dot product <data, data>
    temp = 0
    for i in range(n_trains):
        for j in range(i+1, n_trains):
            temp += Md_dot_product(observed_spike_trains[i], observed_spike_trains[j], delta, dt=dt)

    normalization = n_trains*(n_trains - 1)/2
    dot_product_observed_observed = temp/normalization

    normalization = dot_product_observed_observed + dot_product_predicted_predicted
    MDstar = 2*dot_product_observed_predicted/normalization

    print("Md* = {}".format(MDstar))
    return MDstar


def getSpikeTrain(spiking_times, TMAX, dt):
    """
    Generates a spike spike train graph from spiking times.
    """
    steps = int(TMAX/dt)

    # Convert spiking times to bin indices of size dt.
    spiking_times = np.array(spiking_times, dtype='double')
    spiking_times = spiking_times/dt
    spiking_times = np.array(spiking_times, dtype='int')

    # Generate spike_train graph
    spike_train = np.zeros(steps)
    spike_train[spiking_times] = 1

    return spike_train


def getAverageSpikeTrain(observed_spike_times, TMAX, dt):
    """
    Given a set of spike trains calculates the mean spike train.
    """

    steps = int(TMAX/dt)
    average_spike_train = np.zeros(steps)
    n_spike_trains = len(observed_spike_times)

    for spike_train in observed_spike_times:

        # Convert spiking times to bin indices of size dt.
        spike_train = np.array(spike_train, dtype='double')
        spike_train = spike_train/dt
        spike_train = np.array(spike_train, dtype='int')

        average_spike_train[spike_train] += 1.0

    average_spike_train = average_spike_train/n_spike_trains
    return average_spike_train

score = computeMD(delta, dt)
print(score)
