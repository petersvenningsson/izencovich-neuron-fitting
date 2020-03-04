# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import copy
# Third-party
import pytest
# Local
from models.izencovich_neuron import IzencovichModel
from GA.encoders import RealNumberEncoder
from dataset.dataset_QSNMC import QSNMCDataset
from GA.selection import TournamentCrossover
from GA.population import Population
def test_selector():
    selector = TournamentCrossover()
    parameter_1 = {'a':1,'b':1,'c':1,'d':1}
    parameter_2 = {'a':2,'b':2,'c':2,'d':2}
    dataset = QSNMCDataset()
    NeuronModel = IzencovichModel
    population = Population(NeuronModel, 2, dataset)
    dataset = QSNMCDataset()
    neuron_1 = IzencovichModel(dataset,**parameter_1)
    neuron_2 = IzencovichModel(dataset,**parameter_2)
    population.individuals = [neuron_1, neuron_2]
    individual_1, individual_2 = selector.one_point_crossover(neuron_1, neuron_2, population)
    print(vars(individual_1))
    print(vars(individual_2))
    assert True

def test_encoder_decoder():
    """ Test: RealNumberEncoder can retrieve correct decoded parameters.
    """
    encoder = RealNumberEncoder()
    dataset = QSNMCDataset()

    true_parameters = {'a':0.02,'b':0.2,'c':-65,'d':8}

    neuron = IzencovichModel(dataset,**true_parameters)
    encoded_params = encoder.encode(neuron)
    decoded_parameters = encoder.decode(neuron, **encoded_params)

    for true_value, encoder_decoder_value in zip(true_parameters.values(),decoded_parameters.values()):
        assert true_value == encoder_decoder_value

test_selector()