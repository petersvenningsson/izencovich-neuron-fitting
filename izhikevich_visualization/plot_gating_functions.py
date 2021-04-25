###########
# IMPORTS #
###########
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns
#############
# CONSTANTS #
#############
Ek = -12
ENa = 120
EL = 10.6
V = 1
#############
# FUNCTIONS #
#############


def alpha_m(V):
    alpha = 0.1*(25-V)/(np.exp((25-V)/10) - 1)
    return alpha

def beta_m(V):
    beta_m = 4 * np.exp(-V / 18)
    return beta_m

def alpha_h(V):
    alpha_h = 0.07 * np.exp(-V / 20)
    return alpha_h

def beta_h(V):
    beta_h = 1 / (np.exp((30 - V) / 10) + 1)
    return beta_h

def mdot(t, m):
    m_dot = alpha_m(V)*(1-m) - beta_m(V)*m
    return m_dot

def hdot(t, h):
    hdot = alpha_h(V)*(1 - h) - beta_h(V)*h
    return hdot

###########
# SCRIPTS #
###########
sns.set(style="dark")
fig, ax1 = plt.subplots(figsize=(12, 4))

tspan = [0, 100]
t_eval = np.linspace(0, 100, 50)
voltages = np.linspace(-40, 100, 50)
solutions = np.zeros((voltages.shape[0], 50))
for i, V in enumerate(voltages):
    sol = integrate.solve_ivp(mdot, tspan, [0.2], t_eval = t_eval)
    solutions[i,:] = sol.y
ax1.plot(voltages, solutions[:,-1], 'tab:blue', label=r'$m_\infty$', alpha = 0.5)

tspan = [0, 100]
t_eval = np.linspace(0, 100, 50)
voltages = np.linspace(-40, 100, 50)
solutions = np.zeros((voltages.shape[0], 50))
for i, V in enumerate(voltages):
    sol = integrate.solve_ivp(hdot, tspan, [0.2], t_eval = t_eval)
    solutions[i,:] = sol.y
ax1.plot(voltages, solutions[:,-1], 'tab:green', label=r'$h_\infty$', alpha = 0.5)
fig.tight_layout()
ax1.set_xlabel('V (mV)',fontsize=22)
ax1.legend(loc=4, prop={'size': 22})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(right=1)
plt.show()
fig.savefig("gating.jpg")

