# Author
# Peter Svenningsson
# Email
# peter.o.svenningsson@gmail.com
###########
# IMPORTS #
###########
# Standard
import json

# Third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Local
sns.set(style="dark")
#df = pd.read_csv('mean_fitness_seeded.txt', header=None)
df = pd.read_csv('mean_fitness.txt', header=None)
numpy_array = df.to_numpy().squeeze()
plt.plot(range(60), numpy_array, 'tab:blue', label='mean fitness', alpha = 0.9)
#df = pd.read_csv('max_fitness_seeded.txt', header=None)
df = pd.read_csv('max_fitness.txt', header=None)
numpy_array = df.to_numpy().squeeze()
plt.plot(range(60), numpy_array, 'tab:red', label='maximum fitness', alpha = 0.9)
plt.legend(loc=4)
plt.ylim(0.99, 1)
plt.xlabel('Generation')
plt.ylabel("Fitness")
plt.savefig('fitness_dynamics.jpg')