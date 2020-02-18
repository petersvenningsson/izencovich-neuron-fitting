###########
# IMPORTS #
###########
# standard
from abc import ABC, abstractmethod

###########
# CLASSES #
###########


class Neuron(ABC):
    """ Abstract class for a neuron model. Extends Abstract Base Classes.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def set_parameters(self, **kwargs):
        pass

    @abstractmethod
    def simulate_spiking_times(self):
        pass

    @abstractmethod
    def get_fitness(self):
        pass

    @abstractmethod
    def get_spike_times(self):
        pass