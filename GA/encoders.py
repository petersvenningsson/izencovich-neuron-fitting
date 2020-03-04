# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# standard
from abc import ABC, abstractmethod
import random
###########
# CLASSES #
###########


class Encoder(ABC):
    """ Abstract class for a encoder.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, **kwargs):
        pass

    @abstractmethod
    def decode(self, **kwargs):
        pass

class RealNumberEncoder(Encoder):
    """ A real number encoder encodes the parametrization of a individual to numbers in [0,1]. """
    def __init__(self):
        pass

    def encode(self, neuron):
        """ Encodes the parameters of the input neuron to the interval [0, 1].
         INPUT:
            neuron: A neuron model which interfaces Neuron.
        OUTPUT:
            encoded_parameters: dictionary of encoded parameters.
        """

        encoded_parameters = {}
        for parameter, interval_string in neuron.parameter_intervals.items():
            parameter_value = getattr(neuron, parameter)

            # Extract float type objects for parameter interval bounds.
            lower_bound, upper_bound = [float(s) for s in interval_string.split(':')]
            interval_size = upper_bound - lower_bound

            encoded_parameters[parameter] = (parameter_value - lower_bound)/(interval_size)
        return encoded_parameters

    def decode(self, neuron, **kwargs):
        """ Decodes the parameters of the input neuron to the interval defined by <Neuron>.json.
         INPUT:
            neuron: A neuron model which interfaces Neuron.
            **kwargs: Keyworded encoded parameters.
        OUTPUT:
            encoded_parameters: dictionary of encoded parameters.
        """

        decoded_parameters = {}
        for parameter, encoded_parameter_value in kwargs.items():
            # Extract float type objects for parameter interval bounds.
            lower_bound, upper_bound = [float(s) for s in neuron.parameter_intervals[parameter].split(':')]
            interval_size = upper_bound - lower_bound
            decoded_parameters[parameter] = encoded_parameter_value*interval_size + lower_bound
        return decoded_parameters

    @staticmethod
    def get_initialized_encoding():
        """ Returns a random number in [0,1]
        """
        return random.random()

if __name__ == "main":
    encode = RealNumberEncoder()
