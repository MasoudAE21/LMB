# post/streamfunction.py

import numpy as np


def streamfunction(u):
    """
    Compute streamfunction from a 2D velocity field.

    u[0] = ux
    u[1] = uy
    """

    ux = u[0]
    uy = u[1]

    ny, nx = ux.shape

    dx = 1.0 / nx
    dy = 1.0 / ny

    # ---------------------------------
    # From:
    #
    # ux = d(psi)/dy
    # ---------------------------------

    psi_y = np.zeros_like(ux)

    psi_y[1:, :] = np.cumsum(
        0.5
        * (
            ux[1:, :]
            + ux[:-1, :]
        )
        * dy,
        axis=0
    )

    # ---------------------------------
    # From:
    #
    # uy = -d(psi)/dx
    # ---------------------------------

    psi_x = np.zeros_like(uy)

    psi_x[:, 1:] = -np.cumsum(
        0.5
        * (
            uy[:, 1:]
            + uy[:, :-1]
        )
        * dx,
        axis=1
    )

    # For an incompressible field these
    # should be approximately equal.
    psi = 0.5 * (
        psi_x + psi_y
    )

    return psi