 mutation_probability=0.25, gaussian_creep=True, creep_rate = 0.01)
_(self, crossover_probability = 0.7, tournament_selection_parameter = 0.7, tournament_size = 3):¨

  {
    "_comment": "This file contains the parametrization of the Izencovich neuron. The max and min values of parameters must be set. Any parameters included in 'parameter_intervals' will be used in the optimization. The parameter must be found in the attribute of the Neuron model.",
    "parameter_intervals": {
      "a": "-0.1:0.4",
      "b": "0:2",
      "c": "-65:-50",
      "d": "-2:30"
    },
    "neuron_seeds": {
      "Regular Spiking": {
        "a": "0.02",
        "b": "0.2",
        "c": "-65",
        "d": "8"
      },
      "Chattering": {
        "a": "0.02",
        "b": "0.2",
        "c": "-50",
        "d": "2"
      },
      "Fast Spiking": {
        "a": "0.1",
        "b": "0.2",
        "c": "-65",
        "d": "2"
      },
      "Intrinsically Bursting": {
        "a": "0.02",
        "b": "0.2",
        "c": "-55",
        "d": "4"
      },
      "Thalamo-cortical": {
        "a": "0.02",
        "b": "0.25",
        "c": "-65",
        "d": "0.05"
      },
      "Resonator": {
        "a": "0.1",
        "b": "0.26",
        "c": "-65",
        "d": "2"
      },
      "Low-threshold Spiking": {
        "a": "0.02",
        "b": "0.25",
        "c": "-65",
        "d": "2"
      },
      "spike frequency adaption":{
        "a": "0.01",
        "b": "0.2",
        "c": "-55",
        "d": "4"
      }
    }
  }
