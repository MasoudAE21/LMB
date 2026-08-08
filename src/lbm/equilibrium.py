import numpy as np
from lbm.lattice import *


def _weights_for_field(lattice, ndim):
    """
    Reshape lattice weights for broadcasting over a spatial field.

    Example
    -------
    rho.shape = (ny, nx)
    result.shape = (Q, 1, 1)
    """
    return lattice.w.reshape((lattice.Q,) + (1,) * ndim)


def flow_equilibrium(rho, u, lattice):
    """
    Compute the second-order equilibrium distribution
    for the hydrodynamic LBM.

    Intended primarily for D2Q9.

    Parameters
    ----------
    rho : ndarray
        Density field.

        D2:
            shape = (ny, nx)

    u : ndarray
        Velocity field.

        D2:
            shape = (2, ny, nx)

        u[0] = ux
        u[1] = uy

    lattice : lattice class
        Normally D2Q9.

    Returns
    -------
    feq : ndarray
        Equilibrium distribution.

        shape = (Q, *rho.shape)
    """

    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    if u.shape[0] != lattice.D:
        raise ValueError(
            f"Expected velocity with first dimension "
            f"{lattice.D}, got {u.shape[0]}."
        )

    if u.shape[1:] != rho.shape:
        raise ValueError(
            "Velocity spatial shape must match rho shape."
        )

    # c_i dot u
    #
    # lattice.c : (Q, D)
    # u         : (D, ...)
    # cu        : (Q, ...)
    cu = np.einsum(
        "id,d...->i...",
        lattice.c,
        u
    )

    # |u|^2
    u2 = np.sum(u**2, axis=0)

    w = _weights_for_field(
        lattice,
        rho.ndim
    )

    cs2 = lattice.cs2
    cs4 = lattice.cs4

    feq = (
        w
        * rho[None, ...]
        * (
            1.0
            + cu / cs2
            + 0.5 * cu**2 / cs4
            - 0.5 * u2[None, ...] / cs2
        )
    )

    return feq


def scalar_equilibrium(phi, u, lattice):
    """
    Parameters
    ----------
    phi : ndarray
        Scalar field.

        D1:
            shape = (nx,)

        D2:
            shape = (ny, nx)

    u : ndarray or None
        Velocity field.

        D1:
            shape = (1, nx)

        D2:
            shape = (2, ny, nx)

        For pure diffusion, u may be None.

    lattice : lattice class
        DxQy

    Returns
    -------
    geq : ndarray
        Scalar equilibrium populations.

        shape = (Q, *phi.shape)
    """

    phi = np.asarray(
        phi,
        dtype=np.float64
    )

    w = _weights_for_field(
        lattice,
        phi.ndim
    )

    # Pure diffusion:
    #
    # g_i^eq = w_i * phi
    if u is None:
        return (
            w
            * phi[None, ...]
        )

    u = np.asarray(
        u,
        dtype=np.float64
    )

    if u.shape[0] != lattice.D:
        raise ValueError(
            f"Expected velocity with first dimension "
            f"{lattice.D}, got {u.shape[0]}."
        )

    if u.shape[1:] != phi.shape:
        raise ValueError(
            "Velocity spatial shape must match phi shape."
        )

    # c_i dot u
    cu = np.einsum(
        "id,d...->i...",
        lattice.c,
        u
    )

    # First-order scalar equilibrium
    geq = (
        w
        * phi[None, ...]
        * (
            1.0
            + cu / lattice.cs2
        )
    )

    return geq