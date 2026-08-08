# problems/problem2_transient_conduction.py

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from lbm.lattice.d1q3 import D1Q3

from lbm.collision import collide_scalar
from lbm.initialize import initialize_scalar
from lbm.macroscopic import scalar_macroscopic
from lbm.streaming import stream

from boundary.scalar_halfway import apply_dirichlet


@dataclass
class Problem2Config:
    """
    Configuration for 1D transient heat conduction.
    """

    # Number of fluid lattice nodes
    nx: int = 201

    # Physical rod length
    length: float = 1.0

    # Boundary temperatures
    T_left: float = 1.0
    T_right: float = 0.0

    # LBM thermal relaxation time
    #
    # The project statement does not specify
    # thermal diffusivity/tau, so this remains
    # configurable.
    tau: float = 2.0

    # Maximum iterations
    max_steps: int = 10_000

    # Steady convergence criterion
    tolerance: float = 1.0e-10

    # Required live-plot update interval
    plot_every: int = 50


def analytical_steady_temperature(
    x,
    length,
    T_left,
    T_right
):
    """
    Steady analytical solution of 1D conduction
    with fixed temperatures at both ends.
    """

    return (
        T_left
        + (T_right - T_left)
        * x
        / length
    )


def run_problem2(
    config=None,
    live_plot=True
):
    """
    Solve the 1D transient heat-conduction problem
    using D1Q3 BGK LBM.
    """

    if config is None:
        config = Problem2Config()

    lattice = D1Q3

    if config.nx < 2:
        raise ValueError(
            "nx must be at least 2."
        )

    if config.tau <= 0.5:
        raise ValueError(
            "tau must be greater than 0.5."
        )

    # -------------------------------------------------
    # Grid
    # -------------------------------------------------
    #
    # Halfway bounce-back places the physical walls
    # half a lattice spacing outside the first/last
    # fluid nodes.
    #
    # Therefore fluid nodes are cell centers:
    #
    # x_j = (j + 1/2) * L / Nx
    #
    dx_physical = (
        config.length
        / config.nx
    )

    x = (
        np.arange(
            config.nx,
            dtype=np.float64
        )
        + 0.5
    ) * dx_physical

    # -------------------------------------------------
    # Initial condition
    # -------------------------------------------------
    #
    # T(x,0) = 0 at all interior/fluid points.
    #
    T = np.zeros(
        config.nx,
        dtype=np.float64
    )

    g = initialize_scalar(
        phi=T,
        lattice=lattice,
        u=None
    )

    # -------------------------------------------------
    # LBM thermal diffusivity in lattice units
    # -------------------------------------------------
    alpha_lattice = (
        lattice.cs2
        * (config.tau - 0.5)
    )

    print(
        f"D1Q3 thermal diffusivity "
        f"(lattice units): "
        f"{alpha_lattice:.6g}"
    )

    # -------------------------------------------------
    # Convergence history
    # -------------------------------------------------
    convergence_steps = []
    convergence_values = []

    # -------------------------------------------------
    # Live convergence plot
    # -------------------------------------------------
    if live_plot:

        plt.ion()

        convergence_fig, convergence_ax = (
            plt.subplots(
                figsize=(7, 5)
            )
        )

        (
            convergence_line,
        ) = convergence_ax.semilogy(
            [],
            [],
            marker="o",
            markersize=3
        )

        convergence_ax.set_xlabel(
            "Time step"
        )

        convergence_ax.set_ylabel(
            r"$\max|T^{n+1}-T^n|$"
        )

        convergence_ax.set_title(
            "D1Q3 convergence history"
        )

        convergence_ax.grid(
            True,
            alpha=0.3
        )

    # -------------------------------------------------
    # Main LBM loop
    # -------------------------------------------------
    converged = False

    for step in range(
        1,
        config.max_steps + 1
    ):

        # Macroscopic temperature before update
        T_old = scalar_macroscopic(g)

        # --------------------------
        # Collision
        # --------------------------
        g_post = collide_scalar(
            g=g,
            phi=T_old,
            tau=config.tau,
            lattice=lattice,
            u=None
        )

        # --------------------------
        # Streaming
        # --------------------------
        g_streamed = stream(
            populations=g_post,
            lattice=lattice
        )

        # --------------------------
        # Left Dirichlet wall
        #
        # x = 0
        # T = 1
        # --------------------------
        g_streamed = apply_dirichlet(
            g_streamed=g_streamed,
            g_post=g_post,
            lattice=lattice,
            axis=0,
            side="low",
            value=config.T_left
        )

        # --------------------------
        # Right Dirichlet wall
        #
        # x = L
        # T = 0
        # --------------------------
        g_streamed = apply_dirichlet(
            g_streamed=g_streamed,
            g_post=g_post,
            lattice=lattice,
            axis=0,
            side="high",
            value=config.T_right
        )

        # Advance state
        g = g_streamed

        # New macroscopic temperature
        T = scalar_macroscopic(g)

        # --------------------------
        # Residual
        # --------------------------
        residual = np.max(
            np.abs(
                T - T_old
            )
        )

        # --------------------------
        # Required live update
        # every 50 time steps
        # --------------------------
        if (
            step % config.plot_every == 0
            or step == 1
        ):

            convergence_steps.append(
                step
            )

            convergence_values.append(
                residual
            )

            print(
                f"step = {step:8d}, "
                f"residual = "
                f"{residual:.6e}"
            )

            if live_plot:

                convergence_line.set_data(
                    convergence_steps,
                    convergence_values
                )

                convergence_ax.relim()
                convergence_ax.autoscale_view()

                convergence_fig.canvas.draw_idle()
                convergence_fig.canvas.flush_events()

                plt.pause(0.001)

        # --------------------------
        # Convergence test
        # --------------------------
        if residual < config.tolerance:

            converged = True

            print(
                "\nConverged at "
                f"step {step}"
            )

            print(
                "Final residual = "
                f"{residual:.6e}"
            )

            break

    if not converged:

        print(
            "\nWarning: maximum number "
            "of steps reached."
        )

        print(
            "Final residual = "
            f"{residual:.6e}"
        )

    # -------------------------------------------------
    # Final analytical solution
    # -------------------------------------------------
    T_analytical = (
        analytical_steady_temperature(
            x=x,
            length=config.length,
            T_left=config.T_left,
            T_right=config.T_right
        )
    )

    # -------------------------------------------------
    # Error
    # -------------------------------------------------
    error_linf = np.max(
        np.abs(
            T - T_analytical
        )
    )

    error_l2 = np.sqrt(
        np.mean(
            (
                T
                - T_analytical
            ) ** 2
        )
    )

    print(
        "\nSteady-state errors:"
    )

    print(
        f"L_inf = {error_linf:.6e}"
    )

    print(
        f"L2    = {error_l2:.6e}"
    )

    # -------------------------------------------------
    # Add actual wall points for final plot
    # -------------------------------------------------
    x_plot = np.concatenate(
        (
            [0.0],
            x,
            [config.length]
        )
    )

    T_numerical_plot = np.concatenate(
        (
            [config.T_left],
            T,
            [config.T_right]
        )
    )

    T_analytical_plot = (
        analytical_steady_temperature(
            x=x_plot,
            length=config.length,
            T_left=config.T_left,
            T_right=config.T_right
        )
    )

    # -------------------------------------------------
    # Final temperature profile
    # -------------------------------------------------
    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        x_plot,
        T_analytical_plot,
        label="Analytical steady solution",
        linewidth=2
    )

    plt.plot(
        x_plot,
        T_numerical_plot,
        "o",
        markersize=4,
        markevery=max(
            1,
            config.nx // 20
        ),
        label="D1Q3 numerical"
    )

    plt.xlabel("x")
    plt.ylabel("Temperature")
    plt.title(
        "1D transient conduction: final profile"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    if live_plot:
        plt.ioff()

    plt.show()

    # -------------------------------------------------
    # Return results for tests / postprocessing
    # -------------------------------------------------
    return {
        "x": x,
        "temperature": T,
        "analytical": T_analytical,
        "populations": g,
        "steps": step,
        "residual": residual,
        "convergence_steps": np.asarray(
            convergence_steps
        ),
        "convergence_values": np.asarray(
            convergence_values
        ),
        "error_linf": error_linf,
        "error_l2": error_l2,
        "alpha_lattice": alpha_lattice,
        "converged": converged,
    }


if __name__ == "__main__":

    run_problem2()