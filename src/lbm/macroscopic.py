# solver/macroscopic.py

import numpy as np


def scalar_macroscopic(g):
    """
    Recover scalar field from scalar populations.

    Parameters
    ----------
    g : ndarray
        Scalar distribution.

        D1:
            shape = (Q, nx)

        D2:
            shape = (Q, ny, nx)

    Returns
    -------
    phi : ndarray
        Scalar macroscopic field.
    """

    return np.sum(
        g,
        axis=0
    )


def flow_macroscopic(
    f,
    lattice,
    force=None
):
    """
    Recover density and velocity from flow populations.

    Parameters
    ----------
    f : ndarray
        Flow populations.

        shape = (Q, ny, nx)

    lattice : lattice class
        Normally D2Q9.

    force : ndarray or None
        Optional body-force field.

        shape = (D, ny, nx)

        When supplied, the standard half-force
        velocity correction is included:

            rho*u = sum_i(c_i f_i) + F/2

    Returns
    -------
    rho : ndarray
        Density field.

    u : ndarray
        Velocity field.

        shape = (D, ny, nx)
    """

    rho = np.sum(
        f,
        axis=0
    )

    # Momentum:
    #
    # sum_i c_i f_i
    momentum = np.einsum(
        "id,i...->d...",
        lattice.c,
        f
    )

    if force is not None:

        force = np.asarray(
            force,
            dtype=np.float64
        )

        if force.shape != momentum.shape:
            raise ValueError(
                "force must have the same shape "
                "as the momentum field."
            )

        momentum = (
            momentum
            + 0.5 * force
        )

    u = np.zeros_like(
        momentum,
        dtype=np.float64
    )

    np.divide(
        momentum,
        rho[None, ...],
        out=u,
        where=rho[None, ...] != 0.0
    )

    return rho, u