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
fig, ax1 = plt.subplots(figsize=(12, 4))
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
#df = pd.read_csv('mean_fitness_seeded.txt', header=None)
df = pd.read_csv('mean_fitness.txt', header=None)
numpy_array = df.to_numpy().squeeze()
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
ax1.plot(range(numpy_array.shape[0]), numpy_array, 'tab:blue', label='mean fitness', alpha = 0.9)
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
#df = pd.read_csv('max_fitness_seeded.txt', header=None)
df = pd.read_csv('max_fitness.txt', header=None)
numpy_array = df.to_numpy().squeeze()
ax2 = ax1.twinx()
ax2.plot(range(numpy_array.shape[0]), numpy_array, 'tab:red', label='maximum fitness', alpha = 0.9)
ax2.set_ylabel('Input Current (pA)', color='tab:red')

ax2.tick_params('y', colors='tab:red')
ax1.tick_params('y', colors='tab:blue', labelsize=18)
ax1.tick_params('x', labelsize=18)
# ax2.legend(loc=4, prop={'size': 22})
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
ax2.set_ylim(0.97, 1)
ax2.set_xlabel('Generation',fontsize=22)
ax2.set_ylabel("Maximum fitness", fontsize=22,  color='tab:red')
ax1.set_ylabel("Mean fitness", fontsize=22, color='tab:blue')
ax1.set_xlabel("Generation", fontsize=22)
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

fig.tight_layout()
plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.savefig('fitness_dynamics.jpg')