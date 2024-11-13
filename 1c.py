import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1  
L = 1 
t_max = 5  
dx = 0.05  
dt_list = [0.0005, 0.0001, 0.00125] 
x = np.arange(0, L + dx, dx) 
nx = len(x) 

# Function to compute the analytical solution
def analytical_solution(x, t, N=20):
    u_analytical = np.zeros_like(x)
    for n in range(1, N + 1):
        A_n = (16 * L / (n**3 * np.pi**3)) * (1 - (-1)**n)
        u_analytical += A_n * np.sin(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L)**2 * t)
    return u_analytical

# Plotting results for each dt
plt.figure(figsize=(10, 6))
for dt in dt_list:
    alpha = D * dt / dx**2  
    main_diag = 1 + alpha
    off_diag = -alpha / 2

    # Matrices A and B for Crank-Nicolson
    A = np.diag(main_diag * np.ones(nx)) + np.diag(off_diag * np.ones(nx-1), k=1) + np.diag(off_diag * np.ones(nx-1), k=-1)
    B = np.diag((2 - main_diag) * np.ones(nx)) + np.diag(-off_diag * np.ones(nx-1), k=1) + np.diag(-off_diag * np.ones(nx-1), k=-1)
    A[0, :] = A[-1, :] = 0; A[0, 0] = A[-1, -1] = 1
    B[0, :] = B[-1, :] = 0

    # Initial concentration profile
    u = 4 * x * (1 - x)
    for _ in range(int(t_max / dt) + 1):
        b = B @ u
        b[0] = b[-1] = 0
        u = np.linalg.solve(A, b)
    
    plt.plot(x, u, label=f'Numerical dt={dt:.5f}')

# Analytical solution at t = t_max
u_analytical_at_t = analytical_solution(x, t_max)
plt.plot(x, u_analytical_at_t, label='Analytical Solution at t=5.0s', color='red', linestyle='--')

# Plotting details
plt.xlabel('Position on bar')
plt.ylabel('Concentration')
plt.title('Concentration Distribution at t=5s')
plt.legend()
plt.grid(True)
plt.show()
