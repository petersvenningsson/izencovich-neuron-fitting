# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import random
import copy
# Local
from .mutators import CreepMutation
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
        self.most_fit_individual = None
        self.dataset = dataset

    def initialize_population_seeded(self, encoder):
        """ Initializes the population gaussianly around the seeded models.
        """
        individuals = []
        for i in range(0, self.population_size):
            individual = self.initialize_individual_seeded(encoder)
            individuals.append(individual)
        self.individuals = individuals

    def initialize_individual_seeded(self, encoder):

        # A mutator is created to emulate the gaussian initialization around the seeded models.
        mutator = CreepMutation(encoder, mutation_probability=1, creep_rate=0.035, gaussian_creep=True)
        seed = random.choice(list(self.NeuronModel.neuron_seeds.keys()))
        parameters = self.NeuronModel.neuron_seeds[seed]
        new_individual = self.NeuronModel(self.dataset, **parameters)

        mutator.mutate_individual(new_individual)
        return new_individual



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



    def set_most_fit_individual(self):
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness,reverse = True)
        self.most_fit_individual = copy.copy(sorted_individuals[0])