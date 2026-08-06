# solver/initialize.py

import numpy as np

from solver.equilibrium import (
    flow_equilibrium,
    scalar_equilibrium,
)


def initialize_flow(
    spatial_shape,
    rho0,
    u0,
    lattice
):
    """
    Initialize a flow field at equilibrium.

    Parameters
    ----------
    spatial_shape : tuple
        Example:
            (ny, nx)

    rho0 : float
        Initial density.

    u0 : array-like
        Initial velocity vector.

        Example:
            [0.0, 0.0]

    lattice : lattice class
        Normally D2Q9.

    Returns
    -------
    f, rho, u
    """

    rho = np.full(
        spatial_shape,
        rho0,
        dtype=np.float64
    )

    u0 = np.asarray(
        u0,
        dtype=np.float64
    )

    if u0.shape != (lattice.D,):
        raise ValueError(
            f"u0 must have shape ({lattice.D},)"
        )

    u = np.empty(
        (lattice.D, *spatial_shape),
        dtype=np.float64
    )

    for d in range(lattice.D):

        u[d] = u0[d]

    f = flow_equilibrium(
        rho=rho,
        u=u,
        lattice=lattice
    )

    return f, rho, u


def initialize_scalar(
    phi,
    lattice,
    u=None
):
    """
    Initialize scalar populations at equilibrium.

    Parameters
    ----------
    phi : ndarray
        Initial scalar field.

    lattice : lattice class
        D1Q3 or D2Q5.

    u : ndarray or None
        Velocity field.

    Returns
    -------
    g : ndarray
        Initial scalar populations.
    """

    phi = np.asarray(
        phi,
        dtype=np.float64
    )

    return scalar_equilibrium(
        phi=phi,
        u=u,
        lattice=lattice
    )