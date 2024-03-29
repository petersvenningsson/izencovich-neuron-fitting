###########
# IMPORTS #
###########
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve

Ek = -12
ENa = 120
EL = 10.6
V = 1
gK = 36
gNa = 120
gL = 0.3
I = 0
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

def alpha_n(V):
    alpha_n = 0.01*(10-V)/(np.exp((10-V)/10)-1)
    return alpha_n

def beta_n(V):
    beta_n = 0.125*np.exp(-V/80)
    return beta_n

def m_inf(V):
    m = alpha_m(V)/(alpha_m(V) + beta_m(V))
    return m

def h_inf(V):
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    return h

def tau_n(alpha_n, beta_n):
    tau_n_dot = 1/(alpha_n + beta_n)
    return tau_n_dot

def n_inf(alpha_n, beta_n):
    n_inf = alpha_n/(alpha_n + beta_n)
    return n_inf

def n_dot(V,n):
    n_dot = (n_inf(alpha_n(V), beta_n(V)) - n)/tau_n(alpha_n(V), beta_n(V))
    return n_dot

def V_dot(y):
    V,n = y
    Cvdot = I - gK*n**4*(V-Ek) - gNa*m_inf(V)**3*(0.89 - n) * (V-ENa)- gL*(V-EL)
    return [Cvdot,0]

def _dvdt(V, n):
    n_dot = (n_inf(alpha_n(V), beta_n(V)) - n)/tau_n(alpha_n(V), beta_n(V))
    Cvdot = I - gK*n**4*(V-Ek) - gNa*m_inf(V)**3*(0.89 - n) * (V-ENa)- gL*(V-EL)
    return Cvdot, n_dot

def dvdt(t, y):
    V, n = y
    n_dot = (n_inf(alpha_n(V), beta_n(V)) - n)/tau_n(alpha_n(V), beta_n(V))
    Cvdot = I - gK*n**4*(V-Ek) - gNa*m_inf(V)**3*(0.89 - n) * (V-ENa)- gL*(V-EL)
    return [Cvdot, n_dot]

def normalize(v):
    norm_vec = np.zeros_like(v)
    for i in range(v.shape[1]):
        vec = v[:,i]
        norm = np.linalg.norm(vec)
        if norm == 0:
            norm_vec[:,i] = vec
        else:
            norm_vec[:, i] = vec/norm
    return norm_vec

###########
# SCRIPTS #
###########
sns.set(style="dark")
fig, ax1 = plt.subplots(figsize=(8, 6))




tspan = [0, 15]
t_eval = np.linspace(0, 15, 50)
for j,(initial_n, initial_V) in enumerate(zip([0.4, 0.5, 0.265,0.27,0.3, 0.34, 0.36,0.38,0.42,0.42,0.44,0.44, 0.47,0.4,0.375], [6, 1.3,-2.4,-10,-12, 3,3.5,7,6,6,2, -17,-17.7,-17, -18.5] )):
    print(j)
    sol = integrate.solve_ivp(dvdt, tspan, [initial_V, initial_n], method = "Radau")
    if j == 0:
        ax1.plot(sol.y[0], sol.y[1], 'tab:cyan', alpha = 0.5, label="Trajectories")
    else:
        ax1.plot(sol.y[0], sol.y[1], 'tab:cyan', alpha=0.5)
    derrivatives = _dvdt(sol.y[0], sol.y[1])
    points = np.vstack((sol.y[0], sol.y[1]))
    arrow = np.vstack((derrivatives[0], derrivatives[1]))
    #arrow = normalize(arrow)
    #np.random.seed(24)
    #mask = [ np.random.uniform() > 0.997 for i in range(arrow.shape[1]) ]
    mask = [i % 10 == 0 for i in range(arrow.shape[1])]
    if j in [3,10,2,7,14,15,13]:
        ax1.quiver(points[0, mask], points[1, mask], arrow[0, mask], arrow[1, mask], scale=20000, color="tab:cyan",width =0.001, headwidth= 20, headlength = 10, headaxislength= 7)



# Plot n nullcline
V = np.linspace(-15,120)
n = n_inf(alpha_n(V), beta_n(V))
ax1.plot(V, n, 'darkgreen', alpha = 0.5, linewidth=2, label="n-nullcline")


# plot v nullcline

root = fsolve(V_dot, [0.5,50])
_V = [-1,-1,-1, -1, 0,0,0,0,0, -5,         4,7 ,10,15,20,30,40,30,30,35,40,45, 50, 60, 70, 80, 90, 100,110]
_n = [0.9,0.8, 0.6, 0.5, 0.45,0.44,0.43,0.42, 0.4,0.35,             0.3,0.2, 0.42, 0.45, 0.5, 0.6, 0.65,0.7,0.75,0.75,0.75,0.78,0.78,0.72, 0.65, 0.6, 0.5, 0.4, 0.3]

start_points = [(-10,0.5), (-10,0.4), (-10,0.375), (-8,0.350), (-8,0.325), (0,0.31), (0,0.29), (0,0.28),  (4,0.28),  (5,0.3),  (6,0.31),(6,0.315), (6,0.32), (8,0.325), (8,0.33), (8,0.34), (8,0.35),(10,0.4), (12,0.5)]
solutions = np.zeros((len(start_points),2))
for j, (V,n) in enumerate(start_points):
    root = fsolve(V_dot, [V, n])
    solutions[j,:] = root



from matplotlib.patches import Rectangle
someX, someY = -10, 0.29
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((someX, someY), 22, 0.1, facecolor="grey", alpha=0.4))


ax1.set_xlim(-20,25)
ax1.set_ylim(0.25,0.45)
ax1.plot(solutions[:,0], solutions[:,1], 'tab:red', alpha = 1, linewidth=2, label = "V-nullcline")
ax1.legend(loc=1, prop={'size': 20})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax1.set_xlabel('Potential $V$ (mV)', fontsize= 20)
ax1.set_ylabel('Gating variable $n$', fontsize= 20)
fig.tight_layout()
plt.show()

fig.savefig("stabile_zoom.jpg")
print("a")