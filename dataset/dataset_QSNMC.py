""" Dataset loader for QSNMC 2009a dataset."""
###########
# IMPORTS #
###########
# Standard
import os
import pickle

# Third party
import numpy as np
from NeuroTools.signals import AnalogSignal

#############
# CONSTANTS #
#############
PATH = os.path.dirname(os.path.abspath(__file__))

#############
# CLASSES #
#############

# TODO: create interface for dataset type object
class QSNMCDataset:
    """ Dataset loader for data belonging the 2009 competition QSNMC.
    """
    def __init__(self, filecurrent = PATH + '/2009a/current.txt', filevoltage = PATH + '/2009a/voltage_allrep.txt', dt = 1e-4, holdout_split = 32.7):
        self.filecurrent = filecurrent
        self.filevoltage = filevoltage
        self.dt = dt
        self.current = None
        self.voltage = None # = i.ext in model
        self.spike_trains = [] # used by GA
        self.gold_voltages = []
        self.load_pickled_dataset()
        self.current = self.current*1e-2

        self._generate_spike_trains()


    def _load_dataset(self):
        """ Loads the raw dataset voltage and current.
        """
        print("Loading data from .txt...")
        current = np.loadtxt(self.filecurrent)
        voltage = np.loadtxt(self.filevoltage)

        # trunkate the current signal to the length of the voltage signal.
        current = current[0:voltage.shape[0]]
        self.current = current
        self.voltage = voltage

    def pickle_dataset(self):
        """ Serializes the dataset for faster load time using Pickle.
        """
        self._load_dataset()
        print("Serializing dataset...")
        current_pickle_path = self.filecurrent + '.pickle'
        voltage_pickle_path = self.filevoltage + '.pickle'
        with open(current_pickle_path, 'wb') as f:
            pickle.dump(self.current, f)
        with open(voltage_pickle_path, 'wb') as f:
            pickle.dump(self.voltage, f)

    def load_pickled_dataset(self):
        """ Loads serialized dataset from .pickle file.
        """
        current_pickle_path = self.filecurrent + '.pickle'
        voltage_pickle_path = self.filevoltage + '.pickle'

        try:
            print("Attempting to read dataset from pickled files")
            with open(current_pickle_path, 'rb') as f:
                current = pickle.load(f)
            with open(voltage_pickle_path, 'rb') as f:
                voltage = pickle.load(f)
        except FileNotFoundError:
            print("Serialized data not found...")
            self.pickle_dataset()
            with open(current_pickle_path, 'rb') as f:
                current = pickle.load(f)
            with open(voltage_pickle_path, 'rb') as f:
                voltage = pickle.load(f)
        self.current = current
        self.voltage = voltage

    def _generate_spike_trains(self):
        """ Calculates spike trains from the voltage data. Spike threshold set to 0.
        """
        print("Generating dataset spike trains.")
        for col in range(self.voltage.shape[1]):
            voltage_trial = self.voltage[:, col]
            self.gold_voltages.append(voltage_trial)

            vm_trial = AnalogSignal(voltage_trial, self.dt)
            spike_train = vm_trial.threshold_detection(0)
            self.spike_trains.append(spike_train.spike_times)
