# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Third party
import numpy as np
from scipy.signal import fftconvolve

###########
# CLASSES #
###########

class MDStarComparator():
    """
    Comparator based on match distance metrics. See Improved Similarity Measures for Small Sets of Spike Trains.
    #TODO Include  a stronger description of the metric.
    """

    def __init__(self, dt = 1e-4, TMAX = 39, delta = 1):
        self.dt = dt
        self.delta = delta
        self.TMAX = TMAX

    def _MD_dot_product(self, spike_train_1, spike_train_2):
        rectangular_size = 2 * int(self.delta / self.dt)
        rectangular_window = np.ones(rectangular_size)
        spike_train_1_filtered = fftconvolve(spike_train_1, rectangular_window, mode='same')
        spike_train_2_filtered = fftconvolve(spike_train_2, rectangular_window, mode='same')

        dot_product_value = np.dot(spike_train_1_filtered, spike_train_2_filtered)
        return dot_product_value

    def get_spike_train(self, spiking_times):
        """
        Generates a spike spike train graph from spiking times.
        """
        steps = int(self.TMAX / self.dt)

        # Convert spiking times to bin indices of size dt.
        spiking_times = spiking_times / self.dt
        spiking_times = np.array(spiking_times, dtype='int')

        # Generate spike_train graph
        spike_train = np.zeros(steps)
        spike_train[spiking_times] = 1

        return spike_train

    def get_average_spike_train(self, observed_spike_times):
        """
        Given a set of spike trains calculates the mean spike train.
        """

        steps = int(self.TMAX / self.dt)
        average_spike_train = np.zeros(steps)
        n_spike_trains = len(observed_spike_times)

        for spike_train in observed_spike_times:
            # Convert spiking times to bin indices of size dt.
            spike_train = spike_train/self.dt
            spike_train = np.array(spike_train, dtype='int')

            average_spike_train[spike_train] += 1.0

        average_spike_train = average_spike_train / n_spike_trains
        return average_spike_train

    def evaluate(self, individual, observed_spike_time_trains):
        """ MD* proposed by Richard Naud as a spike train similarity measure.
         See Improved Similarity Measures for Small Sets of Spike Trains, Richard Naud et al. for
         more information."""
        individual.simulate_spiking_times()
        predicted_spike_time_train = individual.spike_times
        observed_spike_trains = []

        n_trains = len(observed_spike_time_trains)

        for spike_times in observed_spike_time_trains:
            spike_train = self.get_spike_train(spike_times)
            observed_spike_trains.append(spike_train)

        observed_average_spike_train = self.get_average_spike_train(observed_spike_time_trains)
        predicted_spike_train = self.get_spike_train(predicted_spike_time_train)

        # Compute dot product <data, model>
        dot_product_observed_predicted = self._MD_dot_product(observed_average_spike_train, predicted_spike_train)

        # Compute dot product <model, model>
        dot_product_predicted_predicted = self._MD_dot_product(predicted_spike_train, predicted_spike_train)

        # Compute dot product <data, data>
        temp = 0
        for i in range(n_trains):
            for j in range(i + 1, n_trains):
                temp += self._MD_dot_product(observed_spike_trains[i], observed_spike_trains[j])

        normalization = n_trains * (n_trains - 1) / 2
        dot_product_observed_observed = temp / normalization

        normalization = dot_product_observed_observed + dot_product_predicted_predicted
        MDstar = 2 * dot_product_observed_predicted / normalization

        print("Md* = {}".format(MDstar))
        return MDstar
