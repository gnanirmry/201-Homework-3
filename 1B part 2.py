import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # length of bar in meters
D = 1.0  # diffusion coefficient in m²/s
dx = 0.05  # spatial step size (mesh spacing)
x = np.arange(0, L + dx, dx)  # spatial grid points
nx = int((L/dx) + 1) # number of spatial points

# Time slices (different dt values)
dt_values = [0.001, 0.00125, 0.0005]
tmax = 5.0  # maximum time in seconds
t = np.arange(0, tmax + 0.1, 0.1)  # time steps for plotting analytical solution

# Analytical solution function (Fourier series expansion)
def analytical_solution(x, t, N=20):
    u_analytical = np.zeros_like(x)
    for n in range(1, N + 1):
        A_n = (16 * L / (n**3 * np.pi**3)) * (1 - (-1)**n)
        u_analytical += A_n * np.sin(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L)**2 * t)
    return u_analytical

# Plot analytical solution at t = 5.0 seconds
plt.figure(figsize=(10, 6))
u_analytical_at_t = analytical_solution(x, tmax, N=20)
plt.plot(x, u_analytical_at_t, label='Analytical Solution at t=5.0s', color='red', linewidth=2)

# Numerical solution using FTCS for different dt values
for dt in dt_values:
    r = D * dt / dx**2
    nt = int(tmax / dt) + 1  # number of time steps
    u = np.zeros((nt, nx))  # concentration matrix
    u[0, :] = 4 * x * (1 - x)  # initial condition
    
    # FTCS Scheme for numerical solution
    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
    
    # Plot the numerical solution at t = 5.0 seconds (last time step)
    u_numerical_at_t = u[-1, :]
    plt.plot(x, u_numerical_at_t, label=f'FTCS dt={dt:.4f}')
    
plt.xlabel('Position (m)')
plt.ylabel('Concentration')
plt.title('Concentration Profile at t = 5.0 seconds')
plt.legend()
plt.grid(True)
plt.show()
