import numpy as np
import matplotlib.pyplot as plt

D = 1  
L = 1 
t_max = 5  
dx = 0.05  
dt_list = [0.0005, 0.0001, 0.00125] 
x = np.arange(0, L + dx, dx) 
nx = int(L / dx) + 1 

concentration_over_time = []

for dt in dt_list:
    alpha = D * dt / dx**2  
    
    maindia = 1 + alpha
    offdia = -alpha / 2

    # Matrices A and B
    A = np.diag(maindia * np.ones(nx)) + \
        np.diag(offdia * np.ones(nx-1), k=1) + \
        np.diag(offdia * np.ones(nx-1), k=-1)

    B = np.diag((2 - maindia) * np.ones(nx)) + \
        np.diag(-offdia * np.ones(nx-1), k=1) + \
        np.diag(-offdia * np.ones(nx-1), k=-1)

    #  (Dirichlet: u(0,t) = u(L,t) = 0)
    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1
    B[0, :] = B[-1, :] = 0

    # Initial concentration profile along the bar
    u = 4 * x * (1 - x)

    # Time-stepping loop
    Nt = int(t_max / dt) + 1  # Number of time steps to reach t_max
    for _ in range(Nt):
        b = B @ u
        b[0] = b[-1] = 0  # Enforcing boundary conditions in b
        u = np.linalg.solve(A, b)

    # Store result for this dt after reaching t_max
    concentration_over_time.append(u)

# Plotting the results for different dt values at t_max
plt.figure(figsize=(10, 6))
for index, u_final in enumerate(concentration_over_time):
    plt.plot(x, u_final, label=f'dt={dt_list[index]}')

def analytical_solution(x, t, N=20):
    u_analytical = np.zeros_like(x)
    for n in range(1, N + 1):
        A_n = (16 * L / (n**3 * np.pi**3)) * (1 - (-1)**n)
        u_analytical += A_n * np.sin(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L)**2 * t)
    return u_analytical

# Plot analytical solution at t = 5.0 seconds
plt.figure(figsize=(10, 6))
u_analytical_at_t = analytical_solution(x, t_max, N=20)
plt.plot(x, u_analytical_at_t, label='Analytical Solution at t=5.0s', color='red')

plt.xlabel('Position on bar')
plt.ylabel('Concentration')
plt.title(f'Concentration at t=5s')
plt.legend()
plt.grid(True)
plt.show()
