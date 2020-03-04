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
# Third-party
import numpy as np
###########
# CLASSES #
###########

# TODO abstract class for mutators

class CreepMutation:
    def __init__(self, encoder: Encoder, mutation_probability = 0.25, creep_rate = 0.025, gaussian_creep = False):
        self.mutation_probability = mutation_probability
        self.creep_rate = creep_rate
        self.encoder = encoder
        self.gaussian_creep = True

    def mutate_individual(self, neuron):
        """ Mutates the parameters of a individual.
        Input:
            A instance which interfaces NeuronModel.
        Output:
        """
        encoded_parameters = self.encoder.encode(neuron)
        for parameter in encoded_parameters.keys():
            if random.random() < self.mutation_probability:
                if self.gaussian_creep:
                    # for gaussian creep we select a standard deviance so that P(|x|>1) = 0.05%.
                    encoded_parameters[parameter] += self.creep_rate*np.random.normal(loc = 0, scale = 0.5)
                else:
                    encoded_parameters[parameter] += self.creep_rate*random.random() - self.creep_rate/2

        decoded_parameters = self.encoder.decode(neuron, **encoded_parameters)
        neuron.set_parameters(**decoded_parameters)
