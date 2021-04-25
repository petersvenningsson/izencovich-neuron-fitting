# Author:
# Peter Svenningsson
# Email:
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# standard
from abc import ABC, abstractmethod
import random
import copy
###########
# CLASSES #
###########


class TournamentCrossover:
    def __init__(self, crossover_probability=0.7, tournament_selection_parameter=0.7, tournament_size=3):
        self.crossover_probability = crossover_probability
        self.tournament_selection_parameter = tournament_selection_parameter
        self.tournament_size = tournament_size

    def tournament_selection(self, population):
        """ Selects one individual in population.
        """
        two_individuals = [random.choice(population.individuals), random.choice(population.individuals)]
        two_individuals.sort(key=lambda x: x.fitness, reverse=True)

        for i in range(self.tournament_size - 1):
            if random.random() < self.tournament_selection_parameter:
                most_fit_candidate = two_individuals[0]
                return most_fit_candidate
            else:
                two_individuals[0] = random.choice(population.individuals)
                two_individuals.sort(key=lambda x: x.fitness, reverse=True)

        least_fit_candidate = two_individuals[1]
        return least_fit_candidate

    def population_crossover(self, population):
        """ Performs crossover on the population.
        Input:
            population: A population of NeuronModel individuals.
        Output:
        """

        next_generation = []
        for _ in range(int(population.population_size/2)):
            individual_1 = self.tournament_selection(population)
            individual_2 = self.tournament_selection(population)

            # deepcopies of individuals are sent so that previous generation of individuals remains unchanged
            # by crossover.
            next_generation_individual_1, next_generation_individual_2 = self.one_point_crossover(
                copy.copy(individual_1), copy.copy(individual_2), population
            )

            next_generation.append(next_generation_individual_1)
            next_generation.append(next_generation_individual_2)
        return next_generation

    def one_point_crossover(self, individual_1, individual_2, population):
        """ Performs 1 point crossover between two individuals.
        """

        model_parameters = population.NeuronModel.parameter_intervals.keys()
        n_parameters = len(model_parameters)
        crossover_point = random.choice(range(n_parameters))
        parameters_model_1 = dict(vars(individual_1))
        parameters_model_2 = dict(vars(individual_2))

        for i_parameter, parameter in enumerate(model_parameters):
            if i_parameter < crossover_point:
                setattr(individual_2, parameter, parameters_model_1[parameter])
                setattr(individual_1, parameter, parameters_model_2[parameter])
            else:
                break
        return individual_1, individual_2
