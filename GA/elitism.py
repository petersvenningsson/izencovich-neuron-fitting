# Author:
# Peter Svenningsson
# Email:
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# standard
import copy
###########
# CLASSES #
###########


class Elitism:
    def __init__(self, elitism_copies = 1):
        self.elitism_copies = 1

    def elitism(self, population):
        for i_elites in range(self.elitism_copies):
            population.individuals[i_elites] = copy.copy(population.most_fit_individual)