# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########

# Third party
import numpy as np
from NeuroTools.signals import AnalogSignal

# Local
from . import abstract_neuron

# TODO: Implement peak finding to set correct value v_spike.

class IzencovichModel(abstract_neuron.Neuron):
    """
    Model of a Izhikevich neuron.
    """

    def __init__(self, a, b, c, d, dataset, v_spike=30.0, TMAX = 39.0):
        self.a = a  # Recovery parameter for variable u. Low a provides long recovery.
        self.b = b  # Sensativity of u to sub-threshhold dynamics of v.
        self.c = c  # Reset value of potential v after action potential spike.
        self.d = d  # Reset value of potential u after action potential spike.
        self.v_spike = v_spike
        self.TMAX = TMAX
        self.u = None # Initialized in simulate method.
        self.v = None # Initialized in simulate method.
        self.i_ext = dataset.current # External input current.
        self.dt = dataset.dt # Sample resolution
        self.spike_times = self.simulate_spiking_times()
        self.fitness = None

    def set_external_current(self, dataset):
        """ Parsing function for a dataset.
        """
        self.i_ext = dataset.current
        self.dt = dataset.dt

    def euler_forward(self, u, v, I):
        """ Calculates one Euler step for the Izhikevich neuron.
        Inputs:
            v - the current voltage
            u - the current recovery
        Returns:
            _v - the proceeding voltage
            _u - the proceeding recovery
        """

        # The constants 0.04, 5, 140 are the one fits all parameters from Simple Model of Spiking Neurons E. Izhikevich.
        # These constants are justified when simulating large networks of neurons.
        # TODO: Parameterise the four constants 0.04, 5, 140.
        _v = v + self.dt * (0.04 * v ** 2 + 5 * v + 140 - u + I)
        _u = u + self.dt * (self.a * (self.b * v - u))
        return _v, _u

    def simulate_voltage(self):
        """ Simulates the pulse train of the Izhikevich neuron by setting self.v_model.
        Inputs:
            T_max - Maximum simulation time in ms
        Outputs:
        """
        steps = int(self.TMAX / self.dt)

        # Initialize variables
        self.u = np.zeros(steps)
        self.v = -65*np.ones(steps)

        # Initial conditions
        self.u[0]=self.b*self.v[0]

        # Given initial conditions and Izhikevich neuron parameters approximate a solution for v(t).
        for t in range(1, steps):
            i_ext = self.i_ext[t - 1]
            v_iteration = self.v[t - 1]
            u_iteration = self.u[t - 1]

            v_iteration_next, u_iteration_next = self.euler_forward(u_iteration, v_iteration, i_ext)

            if v_iteration_next > self.v_spike:
                v_iteration_next = self.c
                u_iteration_next = u_iteration_next + self.d
            self.v[t] = v_iteration_next
            self.u[t] = u_iteration_next

    def simulate_spiking_times(self):
        """ Calculates the spike train produced by the external current.
        Inputs:
            current - Numpy array type instance describing the external current.
        Outputs:
        """

        spike_trains = []
        self.simulate_voltage()
        voltage_trial = self.v
        vm_trial = AnalogSignal(voltage_trial, self.dt)
        spiking_times = vm_trial.threshold_detection(0).spike_times
        self.spike_times = spiking_times
        return spiking_times

    def set_parameters(self, **kwargs):
        """ Sets the model parameters and then simulates new spiking times.
        Input:
            **kwargs: Named parameters for any attribute of model.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.simulate_spiking_times()

    def get_fitness(self):
        return self.fitness

    def get_spike_times(self):
        return self.spike_times