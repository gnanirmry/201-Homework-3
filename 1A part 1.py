import numpy as np
import matplotlib.pyplot as plt

D = 1  # Diffusion coefficient in m^2/s
L = 1  # Length of the bar in m
t_values = np.arange(0, 5.1, 0.1)  # Time points from 0 to 5 in steps of 0.1s
x_values = np.linspace(0, L, 100)  
N_values = [1, 5, 10, 15, 20]  # Number of terms to include in Fourier series

# Initial concentration profile function
def initial_concentration(x,t=0):
    return 4 * x * (1 - x)

# defining Fourier equation
def concentration_profile(x, t, num_terms):
    C_profile = np.zeros_like(x)
    for n in range(1, num_terms + 1):
        An = 16/(np.pi**3*n**3)*(1-(-1)**n)
        C_profile += An * np.sin(n * np.pi * x/L) * np.exp(-D * (n * np.pi)**2 * t)
    return C_profile

#Plotting Concentration evolution over t=5s at midpoint of the bar
midpoint_concentration=[]

for t in t_values:
    concentration_at_midpoint=concentration_profile(L/2,t,N_values[4]) 


midpoint_concentration = np.array([concentration_profile(L/2, t, N_values[4]) for t in t_values])
midpoint_concentration[0]=initial_concentration (x=L/2, t=0)

plt.plot(t_values, midpoint_concentration,marker='o')
plt.xlabel("Time in sec")
plt.ylabel("Evolution of concentration at midpoint")
plt.title("Concentration Profile at L/2")
plt.legend()
plt.grid(True)
plt.show()
