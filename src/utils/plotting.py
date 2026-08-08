# post/plotting.py

import numpy as np
import matplotlib.pyplot as plt

from utils.streamfunction import streamfunction


def plot_results(result):

    T = result["T"]
    u = result["u"]
    Ra = result["Ra"]

    ux = u[0]
    uy = u[1]

    ny, nx = T.shape

    # Halfway lattice-node coordinates
    x = (
        np.arange(nx) + 0.5
    ) / nx

    y = (
        np.arange(ny) + 0.5
    ) / ny

    X, Y = np.meshgrid(
        x,
        y
    )

    speed = np.sqrt(
        ux**2 + uy**2
    )

    psi = streamfunction(u)

    # =================================
    # 1. Temperature / isotherms
    # =================================

    plt.figure(
        figsize=(6, 5)
    )

    filled = plt.contourf(
        X,
        Y,
        T,
        levels=30
    )

    plt.contour(
        X,
        Y,
        T,
        levels=15
    )

    plt.colorbar(
        filled,
        label="Temperature"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Temperature contours - Ra = {Ra:.0e}"
    )

    plt.axis("equal")
    plt.tight_layout()

    # =================================
    # 2. Velocity field + streamlines
    # =================================

    plt.figure(
        figsize=(6, 5)
    )

    filled = plt.contourf(
        X,
        Y,
        speed,
        levels=30
    )

    plt.colorbar(
        filled,
        label="Velocity magnitude"
    )

    plt.streamplot(
        x,
        y,
        ux,
        uy,
        density=1.5
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Velocity field - Ra = {Ra:.0e}"
    )

    plt.axis("equal")
    plt.tight_layout()

    # =================================
    # 3. Streamfunction contours
    # =================================

    plt.figure(
        figsize=(6, 5)
    )

    contours = plt.contour(
        X,
        Y,
        psi,
        levels=20
    )

    plt.clabel(
        contours,
        inline=True,
        fontsize=8
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Streamfunction - Ra = {Ra:.0e}"
    )

    plt.axis("equal")
    plt.tight_layout()

    # =================================
    # 4. Centerline velocity profiles
    # =================================

    ix = np.argmin(
        np.abs(x - 0.5)
    )

    iy = np.argmin(
        np.abs(y - 0.5)
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    # ux along vertical centerline
    ax[0].plot(
        ux[:, ix],
        y
    )

    ax[0].axvline(
        0.0,
        linewidth=0.8
    )

    ax[0].set_xlabel(
        r"$u_x$"
    )

    ax[0].set_ylabel(
        "y"
    )

    ax[0].set_title(
        r"$u_x$ at $x=0.5$"
    )

    ax[0].grid()

    # uy along horizontal centerline
    ax[1].plot(
        x,
        uy[iy, :]
    )

    ax[1].axhline(
        0.0,
        linewidth=0.8
    )

    ax[1].set_xlabel(
        "x"
    )

    ax[1].set_ylabel(
        r"$u_y$"
    )

    ax[1].set_title(
        r"$u_y$ at $y=0.5$"
    )

    ax[1].grid()

    fig.suptitle(
        f"Centerline velocity profiles - Ra = {Ra:.0e}"
    )

    fig.tight_layout()

    # =================================
    # Simple numerical diagnostics
    # =================================

    dx = 1.0 / nx
    dy = 1.0 / ny

    du_dx = np.gradient(
        ux,
        dx,
        axis=1
    )

    dv_dy = np.gradient(
        uy,
        dy,
        axis=0
    )

    divergence = (
        du_dx + dv_dy
    )

    print()
    print("-----------------------------")
    print("RESULT DIAGNOSTICS")
    print("-----------------------------")

    print(
        f"T min       = {T.min():.6e}"
    )

    print(
        f"T max       = {T.max():.6e}"
    )

    print(
        f"ux min/max  = "
        f"{ux.min():.6e} / "
        f"{ux.max():.6e}"
    )

    print(
        f"uy min/max  = "
        f"{uy.min():.6e} / "
        f"{uy.max():.6e}"
    )

    print(
        f"max speed   = "
        f"{speed.max():.6e}"
    )

    print(
        f"max |div u| = "
        f"{np.max(np.abs(divergence)):.6e}"
    )

    print(
        f"psi min/max = "
        f"{psi.min():.6e} / "
        f"{psi.max():.6e}"
    )

    print("-----------------------------")
    print()

    plt.show()

    return psi