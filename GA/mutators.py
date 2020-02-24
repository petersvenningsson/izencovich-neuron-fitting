# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
from .encoders import Encoder
import random

###########
# CLASSES #
###########

# TODO abstract class for mutators

class CreepMutation:
    def __init__(self, encoder: Encoder, mutation_probability = 0.1, creep_rate = 0.1):
        self.mutation_probability = mutation_probability
        self.creep_rate = creep_rate
        self.encoder = encoder

    def mutate_individual(self, neuron):
        """ Mutates the parameters of a individual.
        Input:
            A instance which interfaces NeuronModel.
        Output:
        """
        encoded_parameters = self.encoder.encode(neuron)
        for parameter in encoded_parameters.keys():
            if random.random() < self.mutation_probability:
                encoded_parameters[parameter] += self.creep_rate*random.random() - self.creep_rate/2

        decoded_parameters = self.encoder.decode(neuron, **encoded_parameters)
        neuron.set_parameters(**decoded_parameters)
