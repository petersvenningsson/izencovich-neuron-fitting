# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import json

# Third party
import numpy as np
from NeuroTools.signals import AnalogSignal

# Local
from .abstract_neuron import Neuron

# TODO: Implement peak finding to set correct value v_max.

class IzencovichModelExtended(Neuron):
    """
    Model of a Izhikevich neuron.
    """
    with open("models/izencovich_neuron_extended.json", 'r') as file:
        config = json.loads(file.read())
        parameter_intervals = config['parameter_intervals']
        neuron_seeds = config['neuron_seeds']

    def __init__(
            self, dataset, a=None, b=None, c=None, d=None, v_rest = None, v_threshold = None, k = None, C = None,
            current_a=1, v_max=35.0, polynomial_a=0, polynomial_b=0, TMAX = 39.0, spike_times = None, fitness = None,
    ):
        # e = 0.04, g = 5, h = 140
        self.a = float(a)  # Recovery parameter for variable u. Low a provides long recovery. [1/ms]
        self.b = float(b)  # Sensativity of u to sub-threshhold dynamics of v. [pA.1/mV]
        self.c = float(c)  # Reset value of potential v after action potential spike. [mV]
        self.d = float(d)  # Reset value of potential u after action potential spike. [pA]
        self.v_rest = float(v_rest) # resting membrane potential.  [mV]
        self.v_threshold = float(v_threshold) # instanteneous threshold potential. [mV]
        self.k = float(k) # Constant [pA./mV]
        self.C = float(C) # The membrane capacitance [pF]
        self.current_a = float(current_a)
        self.polynomial_a = float(polynomial_a)
        self.polynomial_b = float(polynomial_b)
        self.v_max = float(v_max) # Peak potential. [mV]
        self.TMAX = float(TMAX) # Maximum simulation time [s]
        self.u = None # Initialized in simulate method. Membrane potential [mV]
        self.v = None # Initialized in simulate method. Recovery variable [pA]
        self.dataset = dataset # Save dataset to simplify __copy__
        self.i_ext = dataset.current # External input current. [pA]
        self.dt = dataset.dt # Sample resolution [s]
        self.spike_times = spike_times
        self.fitness = fitness

    def __copy__(self):
        new_copy = IzencovichModelExtended(self.dataset, self.a, self.b, self.c, self.d, self.v_rest, self.v_threshold, self.k,
                                   self.C, self.v_max, self.current_a, self.polynomial_a, self.polynomial_b, self.TMAX, spike_times = self.spike_times, fitness = self.fitness)
        return new_copy


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
        # Time unit of .dt is [s]. Differential equations are defined in unit [ms], hence a constant factor 1000.
        _v = v + 1000*self.dt * 1/self.C*(self.k*(v - self.v_rest)*(v-self.v_threshold) - u + self.current_a*I*10) + self.polynomial_b *((u)**2)
        _u = u + 1000*self.dt * (self.a * (self.b * (v - self.v_rest) - u))
        return _v, _u

    def simulate_voltage(self):
        """ Simulates the pulse train of the Izhikevich neuron by setting self.v_model.
        Inputs:
            T_max - Maximum simulation time in ms
        Outputs:
        """
        #steps = int(self.TMAX / self.dt)

        steps = len(self.i_ext)

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

            if v_iteration_next > self.v_max:
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
        self.spike_times = None
        #self.fitness = None

    def get_fitness(self):
        return self.fitness

    def get_spike_times(self):
        return self.spike_times


class SimpleIzencovichModel(Neuron):
    """
    Model of a Izhikevich neuron.
    """

    with open("models/izencovich_neuron.json", 'r') as file:
        config = json.loads(file.read())
        parameter_intervals = config['parameter_intervals']
        neuron_seeds = config['neuron_seeds']

    def __init__(
            self, dataset, a=None, b=None, c=None, d=None, v_max=35.0, TMAX = 39, spike_times = None, fitness = None, *vararg, **kwargs
    ):
        self.a = float(a)  # Recovery parameter for variable u. Low a provides long recovery. [1/ms]
        self.b = float(b)  # Sensativity of u to sub-threshhold dynamics of v. [pA.1/mV]
        self.c = float(c)  # Reset value of potential v after action potential spike. [mV]
        self.d = float(d)  # Reset value of potential u after action potential spike. [pA]
        self.v_max = float(v_max) # Peak potential. [mV]
        self.TMAX = float(TMAX) # Maximum simulation time [s]
        self.u = None # Initialized in simulate method. Membrane potential [mV]
        self.v = None # Initialized in simulate method. Recovery variable [pA]
        self.dataset = dataset # Save dataset to simplify __copy__
        self.i_ext = dataset.current # External input current. [pA]
        self.dt = dataset.dt # Sample resolution [s]
        self.spike_times = spike_times
        self.fitness = fitness
        self.current_a = 0
        self.v_rest = 0
    def __copy__(self):
        new_copy = SimpleIzencovichModel(self.dataset, self.a, self.b, self.c, self.d,
                self.v_max, self.TMAX, spike_times = self.spike_times, fitness = self.fitness)
        return new_copy


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
        _v = v + 1000 * self.dt * (0.04 * v ** 2 + 5 * v + 140 - u + I*10)
        _u = u + 1000 * self.dt * (self.a * (self.b * v - u))
        return _v, _u

    def simulate_voltage(self):
        """ Simulates the pulse train of the Izhikevich neuron by setting self.v_model.
        Inputs:
            T_max - Maximum simulation time in ms
        Outputs:
        """
        #steps = int(self.TMAX / self.dt)

        steps = len(self.i_ext)

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

            if v_iteration_next > self.v_max:
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
        self.spike_times = None
        #self.fitness = None

    def get_fitness(self):
        return self.fitness

    def get_spike_times(self):
        return self.spike_times