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


    def initialize_population(self):
        """ # TODO docstring"""
        individuals = []
        for i in range(0, self.population_size):
            print(i)
            a = 0.02
            b = 0.2
            c = -50
            d = 2
            individual = self.NeuronModel(a, b, c, d, self.dataset)
            individuals.append(individual)
        self.individuals = individuals