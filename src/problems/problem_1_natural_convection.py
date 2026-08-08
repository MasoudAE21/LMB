# problems/problem1_natural_convection.py

import numpy as np

from lbm.lattice.d2q9 import D2Q9
from lbm.lattice.d2q5 import D2Q5

from lbm.equilibrium import (
    flow_equilibrium,
    scalar_equilibrium
)

from lbm.collision import (
    collide_flow,
    collide_scalar
)

from lbm.streaming import stream

from lbm.macroscopic import (
    flow_macroscopic,
    scalar_macroscopic
)

from lbm.forcing import (
    buoyancy_force,
    guo_source
)

from boundary.flow import no_slip_walls

from boundary.scalar_halfway import (
    dirichlet_left,
    dirichlet_right,
    adiabatic_bottom,
    adiabatic_top
)


def run(
    Ra=1e3,
    nx=101,
    ny=101,
    max_steps=6000,
    tolerance=1e-8
):

    # ---------------------------------
    # Physical / dimensionless values
    # ---------------------------------

    Pr = 0.71

    T_hot = 1.0
    T_cold = 0.0

    T_ref = (
        T_hot + T_cold
    ) / 2.0

    delta_T = (
        T_hot - T_cold
    )

    rho0 = 1.0

    # ---------------------------------
    # Lattice parameters
    # ---------------------------------

    # With halfway walls, nx fluid nodes
    # represent a cavity width of nx
    # lattice spacings.
    L = float(nx)

    Ma = 0.1

    cs = np.sqrt(
        D2Q9.cs2
    )

    U_char = (
        Ma * cs
    )

    # From:
    #
    # Ra = g beta dT L^3 / (nu alpha)
    # Pr = nu / alpha

    nu = (
        U_char
        * L
        * np.sqrt(
            Pr / Ra
        )
    )

    alpha = (
        nu / Pr
    )

    tau_f = (
        0.5
        + nu / D2Q9.cs2
    )

    tau_T = (
        0.5
        + alpha / D2Q5.cs2
    )

    g_beta = (
        U_char**2
        / (
            L
            * delta_T
        )
    )

    print()
    print("-----------------------------")
    print(f"Ra      = {Ra:.3e}")
    print(f"Pr      = {Pr}")
    print(f"nu      = {nu:.6e}")
    print(f"alpha   = {alpha:.6e}")
    print(f"tau_f   = {tau_f:.6f}")
    print(f"tau_T   = {tau_T:.6f}")
    print(f"g_beta  = {g_beta:.6e}")
    print("-----------------------------")
    print()

    # ---------------------------------
    # Initial flow
    # ---------------------------------

    rho = np.full(
        (ny, nx),
        rho0
    )

    u = np.zeros(
        (2, ny, nx)
    )

    f = flow_equilibrium(
        rho,
        u,
        D2Q9
    )

    # ---------------------------------
    # Initial temperature
    #
    # Linear from hot left wall
    # to cold right wall.
    # ---------------------------------

    x = (
        np.arange(nx) + 0.5
    ) / nx

    T = (
        T_hot
        + (T_cold - T_hot)
        * x
    )

    T = np.tile(
        T,
        (ny, 1)
    )

    g = scalar_equilibrium(
        T,
        u,
        D2Q5
    )

    # ---------------------------------
    # Main loop
    # ---------------------------------

    for step in range(max_steps):

        T_old = T.copy()
        u_old = u.copy()

        # =================================
        # FLOW
        # =================================

        # Density before force calculation
        rho = np.sum(
            f,
            axis=0
        )

        # Boussinesq force
        force = buoyancy_force(
            rho,
            T,
            g_beta,
            T_ref
        )

        # Macroscopic velocity
        rho, u = flow_macroscopic(
            f,
            D2Q9,
            force
        )

        # Guo forcing term
        source = guo_source(
            force,
            u,
            tau_f,
            D2Q9
        )

        # Collision
        f_post = collide_flow(
            f,
            rho,
            u,
            tau_f,
            D2Q9,
            source
        )

        # Streaming
        f = stream(
            f_post,
            D2Q9
        )

        # No-slip walls
        f = no_slip_walls(
            f,
            f_post,
            D2Q9
        )

        # =================================
        # TEMPERATURE
        # =================================

        # Collision
        g_post = collide_scalar(
            g,
            T,
            tau_T,
            D2Q5,
            u
        )

        # Streaming
        g = stream(
            g_post,
            D2Q5
        )

        # Hot left wall
        g = dirichlet_left(
            g,
            g_post,
            T_hot,
            D2Q5
        )

        # Cold right wall
        g = dirichlet_right(
            g,
            g_post,
            T_cold,
            D2Q5
        )

        # Adiabatic bottom
        g = adiabatic_bottom(
            g,
            g_post,
            T,
            D2Q5
        )

        # Adiabatic top
        g = adiabatic_top(
            g,
            g_post,
            T,
            D2Q5
        )

        # New temperature
        T = scalar_macroscopic(
            g
        )

        # =================================
        # Updated velocity for convergence
        # =================================

        rho = np.sum(
            f,
            axis=0
        )

        force = buoyancy_force(
            rho,
            T,
            g_beta,
            T_ref
        )

        rho, u = flow_macroscopic(
            f,
            D2Q9,
            force
        )

        # =================================
        # Convergence
        # =================================

        residual_T = np.max(
            np.abs(
                T - T_old
            )
        )

        residual_u = np.max(
            np.abs(
                u - u_old
            )
        )

        residual = max(
            residual_T,
            residual_u
        )

        if step % 1000 == 0:

            speed = np.sqrt(
                u[0]**2
                + u[1]**2
            )

            print(
                f"step = {step:7d}   "
                f"res = {residual:.3e}   "
                f"umax = {np.max(speed):.3e}"
            )

        if residual < tolerance:

            print()
            print(
                f"Converged after "
                f"{step} steps"
            )

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

        "nu": nu,
        "alpha": alpha,

        "residual": residual,
        "steps": step
    }


if __name__ == "__main__":

    result = run(
        Ra=1e3
    )