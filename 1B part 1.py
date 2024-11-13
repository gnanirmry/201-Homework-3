import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  
D = 1.0 
dx = 0.05  # step size
dt = 0.00125  # time step 
tmax = 5.0  # maximum time in seconds

nx = int(L/dx) + 1
nt = int(tmax/dt) + 1
x = np.linspace(0, L, nx)
t = np.linspace(0, tmax, nt)
r = D*dt/dx**2

x_mid = np.abs(x-L/2).argmin()

u = np.zeros((nt, nx))

# initial condition
u[0,:] = 4*x*(1-x)

# Set boundary conditions
u[:,0] = 0    # left boundary
u[:,-1] = 0   # right boundary

# FTCS scheme
for n in range(0, nt-1):
    for i in range(1, nx-1):
        u[n+1,i] = u[n,i] + r*(u[n,i+1] - 2*u[n,i] + u[n,i-1])

concentration_midpoint=u[:, x_mid]

# Plot concentration evolution with time for each position
# plt.figure(figsize=(10, 6))
#for i in range(nx):
    #position = i*dx
    #plt.plot(t, u[:], label=f'x = {position:.2f}m')
   
plt.plot(t, concentration_midpoint, label=f'x={L/2}')
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.title('Concentration Evolution at L/2')
plt.grid(True)
plt.show()