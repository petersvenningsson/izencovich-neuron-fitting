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


class LIF:

    with open("models/LIF.json", 'r') as file:
        config = json.loads(file.read())
        parameter_intervals = config['parameter_intervals']

    def __init__(self, dataset=None, C=1.0, v_rest=-65.0, g_leak=1.0, v_thresh = -45.0, v_reset = -70.0, v_spike=30.0,
                     TMAX=39.0, spike_times=None, fitness=None,
                 ):
        self.C = C  # Capacitance
        self.v_rest = v_rest  # Resting membrane potential
        self.g_leak = g_leak  # Leak conductance
        self.v_thresh = v_thresh  # Leak conductance
        self.v_reset = v_reset
        self.v_spike = v_spike
        self.dataset = dataset
        self.dt = dataset.dt # Sample resolution [s]
        self.spike_times = spike_times
        self.fitness = fitness
        self.TMAX = TMAX
        self.v = None # Initialized in simulate method. Recovery variable [pA]

    def set_external_current(self, dataset):
        self.i_ext = dataset.current
        self.dt = dataset.dt
        self.steps = len(self.i_ext)

    def integrate(self, T_max=None):
        if T_max is None:
            steps = self.steps
        else:
            steps = int(T_max / self.dt)
        self.v_m = np.zeros(steps)
        self.v_m[0] = self.v_rest
        for t in range(1, steps):
            i_ext = self.i_ext[t - 1]
            v_m = self.v_m[t - 1]
            if v_m == self.v_spike:
                v_m = self.v_reset
            dV = self.dVdt(v_m, i_ext) * self.dt
            if t > 0:
                v_m += dV
            if v_m > self.v_thresh:
                v_m = self.v_spike  # AP height.
            self.v_m[t] = v_m
        self.v = self.v_m

    def dVdt(self, v, i_ext):
        g_leak, v_rest, C = self.g_leak, self.v_rest, self.C
        dVdt = (g_leak * (v_rest - v) + i_ext*100) / C
        # print i_ext,v,dVdt
        return dVdt

    def simulate_spiking_times(self, current=None):
        self.set_external_current(self.dataset)
        spike_trains = []
        for trial in range(5):
            self.integrate(T_max=self.TMAX)
            voltage_trial = self.v_m
            vm_trial = AnalogSignal(voltage_trial, self.dt)
            spike_train = vm_trial.threshold_detection(0).spike_times
            spike_trains.append(spike_train)
            if len(spike_trains) >= 3:  # Don't use all the spike trains.
                break
        self.spike_times = spike_trains[0]
        return spike_trains[0]

    def set_parameters(self, **kwargs):
        """ Sets the model parameters and then simulates new spiking times.
        Input:
            **kwargs: Named parameters for any attribute of model.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.spike_times = None
