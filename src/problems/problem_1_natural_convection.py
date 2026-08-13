import numpy as np
from lbm.lattice.d2q9 import D2Q9
from lbm.lattice.d2q5 import D2Q5
from lbm.equilibrium import *
from lbm.collision import *
from lbm.streaming import stream
from lbm.macroscopic import *
from lbm.forcing import *
from boundary.flow import no_slip_walls
from boundary.scalar import *
def run(
    Ra=1e5,
    nx=101,
    ny=101,
    max_steps=50000,
    tolerance=1e-8,
    visco=0.02, # Assumptions, Source: Book
    rho0=6.0    # Assumptions, Source: Book
):
    # Parameters
    Pr = 0.71
    T_hot = 1.0
    T_cold = 0.0
    T_ref = (T_hot + T_cold) / 2.0
    delta_T = (T_hot - T_cold)
    cs = np.sqrt(D2Q9.cs2)
    
    # Ra = g beta dT L^3 / (visco alpha)
    # Pr = visco / alpha
    alpha = (visco / Pr)
    g_beta = (Ra * visco**2) / (Pr * delta_T * ny**3)
    
    # incompressibility check, based on book example
    vel_propertion = np.sqrt(g_beta * delta_T * ny)
    if vel_propertion > 0.15:
        print(f"sqrt(g.beta.dT.Nx) is {vel_propertion}, which is not acceptable for an incompressible flow")
        return
    
    tau_f = (0.5 + visco / D2Q9.cs2)
    tau_T = (0.5 + alpha / D2Q5.cs2)

    print()
    print("-----------------------------")
    print(f"Ra      = {Ra:.3e}")
    print(f"Pr      = {Pr}")
    print(f"visco   = {visco:.6e}")
    print(f"alpha   = {alpha:.6e}")
    print(f"tau_f   = {tau_f:.6f}")
    print(f"tau_T   = {tau_T:.6f}")
    print(f"g_beta  = {g_beta:.6e}")
    print("-----------------------------")
    print()
    
    # Initial flow and temprature
    rho = np.full((ny, nx), rho0)
    u = np.zeros((2, ny, nx))
    f = flow_equilibrium(rho, u, D2Q9)
    x = (np.arange(nx) + 0.5) / nx
    T = (T_hot + (T_cold - T_hot) * x)
    T = np.tile(T, (ny, 1))
    g = scalar_equilibrium(T, u, D2Q5)

    # Main loop
    for step in range(max_steps):
        T_old = T.copy()
        u_old = u.copy()
        # FLOW
        # Density before force calculation
        rho = np.sum(f, axis=0)
        # Boussinesq force
        force = buoyancy_force(rho, T, g_beta, T_ref)
        # Macroscopic velocity
        rho, u = flow_macroscopic(f, D2Q9, force)
        # Guo forcing term
        source = get_source(force, D2Q9)
        # Collision
        f_post = collide_flow(f, rho, u, tau_f, D2Q9, source)
        # Streaming
        f = stream(f_post, D2Q9)
        # No-slip walls
        f = no_slip_walls(f, f_post, D2Q9)
        
        # TEMPERATURE
        # Collision
        g_post = collide_scalar(g, T, tau_T, D2Q5, u)
        # Streaming
        g = stream(g_post, D2Q5)
        # Hot left wall
        g = dirichlet_left(g, g_post, T_hot, D2Q5)
        # Cold right wall
        g = dirichlet_right(g, g_post, T_cold, D2Q5)
        # Adiabatic bottom
        g = adiabatic_bottom(g, g_post, T, D2Q5)
        # Adiabatic top
        g = adiabatic_top(g, g_post, T, D2Q5)
        
        # New Macroscopic variables
        T = scalar_macroscopic(g)
        rho = np.sum(f, axis=0)
        force = buoyancy_force(rho, T, g_beta, T_ref)
        rho, u = flow_macroscopic(f, D2Q9, force)
        
        # Check Convergence
        residual_T = np.max(np.abs(T - T_old))
        residual_u = np.max(np.abs(u - u_old))
        residual = max(residual_T, residual_u)
        if step % 500 == 0:
            speed = np.sqrt(u[0]**2 + u[1]**2)
            print(
                f"step = {step:7d}   "
                f"res = {residual:.3e}   "
                f"umax = {np.max(speed):.3e}"
            )
        if residual < tolerance:
            print()
            print(f"Converged after {step} steps")
            break
    return {
        "Ra": Ra,
        "Pr": Pr,
        "rho": rho,
        "u": u,
        "T": T,
        "f": f,
        "g": g,
        "tau_f": tau_f,
        "tau_T": tau_T,
        "nu": visco,
        "alpha": alpha,
        "residual": residual,
        "steps": step
    }


# if __name__ == "__main__":

#     result = run(
#         Ra=1e3
#     )