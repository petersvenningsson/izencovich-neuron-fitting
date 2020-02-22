# izencovich-neuron-fitting
Fitting of dyn-sys. neuron models to wild data with stochastic optimization methods.

# Exploratory implementation
* Real number encoding
* Mutation
* Crossover
* Elitism
* Population
* Tournament selection

# Abstract classes
Neurons: (Parent of IzencovichNeurons, integrate-and-fire ect)

Encoders: (Real number, Binary)

Evaluator: (MD_star)

# Classes

Population

MutationOperator

CrossOverOperator

IzencovichNeuron

IntegrateAndFireNeuron

MDStarComparator

QSNMCDataset

TournamentSelection

# Config


    <Neuron>.json: 
  
  
defines parametrization of neuron with intervals for random initialization or defines parameters for seeded initialization.

