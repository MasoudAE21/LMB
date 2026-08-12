import numpy as np
import matplotlib.pyplot as plt
from lbm.lattice.d1q3 import D1Q3
from lbm.equilibrium import scalar_equilibrium
from lbm.collision import collide_scalar
from lbm.streaming import stream
from lbm.macroscopic import scalar_macroscopic
from boundary.scalar_halfway import (
    dirichlet_left,
    dirichlet_right
)


def run(thermal_diffusivity=0.5, nx=101, max_steps=20000, tol=1e-6):
    # Problem parameters
    L = 1.0
    T_left = 1.0
    T_right = 0.0
    plot_every = 50
    lattice = D1Q3
    tau = 3.0 * thermal_diffusivity + 0.5
    # Grid
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    # Initial condition
    T = np.zeros(nx)
    g = scalar_equilibrium(T, None, lattice)
    
    # Live convergence plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.semilogy([], [])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Residual")
    ax.set_title("Convergence")
    steps_history = []
    residual_history = []
    
    # Time loop
    for step in range(max_steps):
        T_old = scalar_macroscopic(g)
        # Collision
        g_post = collide_scalar(g, T_old, tau, lattice)
        # Streaming
        g = stream(g_post, lattice)
        # Boundary conditions
        g = dirichlet_left(g, g_post, T_left, lattice)
        g = dirichlet_right(g, g_post, T_right, lattice)
        # New temperature
        T = scalar_macroscopic(g)
        # Convergence
        residual = np.max(np.abs(T - T_old))
        # Live update every 50 steps
        if step % plot_every == 0:
            steps_history.append(step)
            residual_history.append(residual)
            line.set_data(steps_history, residual_history)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)
            print(step, residual)
        if residual < tol:
            print(f"Converged after {step} steps")
            break
    plt.ioff()
   
    # Final temperature plot 
    # Analytical steady solution
    T_exact = (T_left + (T_right - T_left) * x / L)
    plt.figure()
    plt.plot(x, T_exact, label="Analytical")
    plt.plot(x, T, "o", markersize=3, label="LBM")
    plt.xlabel("x")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid()
    plt.show()


# if __name__ == "__main__":
#     run()