import numpy as np
from matplotlib import pyplot as plt

D = 1
L = 1 # Length of the bar in m
N_values = [1,5,10,15,20]
t = 5 # time in seconds
x = np.linspace(0, L, 100)

def concentration_profile(x, t, N):
    Analytical_solution = 0
    for n in range(1, N+1):  
        An = 16/((np.pi**3)*(n**3))*(1-(-1)**n)
        Analytical_solution += An * np.sin(n * np.pi * x) * np.exp(-D * (n * np.pi)**2 * t)
    return Analytical_solution 


# plotting for different values of N
plt.figure(figsize=(10,6))
for N in N_values:
    c_N = concentration_profile(x, t=5.0, N=N)
    plt.plot(x, c_N, label=f'N={N}')

plt.xlabel("Position on the bar(m)")
plt.ylabel("Concentration")
plt.title("Analytical solution at different N")
plt.legend()
plt.grid(True)
plt.show()