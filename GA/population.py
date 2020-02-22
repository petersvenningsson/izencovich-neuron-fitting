# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import random

###########
# CLASSES #
###########
class Population:
    """ Models a population of individuals. #TODO Write must implement specifications.
    """
    def __init__(self, NeuronModel, population_size, dataset):
        self.individuals = []
        self.population_size = population_size
        self.NeuronModel = NeuronModel
        self.best_individual = None
        self.dataset = dataset

    def initialize_population(self, encoder):
        """ Initialize population without any seeding models. Model parameters are randonly selected within
        the config model parameter interval.
        """

        individuals = []
        for i in range(0, self.population_size):
            individual = self.initialize_individual(encoder)
            individuals.append(individual)
        self.individuals = individuals

    def initialize_individual(self, encoder):
        """ Initializes a individual of <NeuronModel> with parameter intervals as specified by <NeuronModel>.json."""

        initialized_encodings = {}
        for parameter in self.NeuronModel.parameter_intervals.keys():
            initialized_encodings[parameter] = encoder.get_initialized_encoding()
        encoded_parameters = encoder.decode(self.NeuronModel, **initialized_encodings)
        individual = self.NeuronModel(self.dataset, **encoded_parameters)

        return individual

