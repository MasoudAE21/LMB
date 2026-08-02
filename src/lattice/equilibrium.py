import numpy as np


def equilibrium(rho, u, lattice):
    """
    General equilibrium distribution.

    Parameters
    ----------
    rho : ndarray
        Density (or concentration).

    u : ndarray
        Velocity field with shape (D, ...).
        Examples:
            D1 -> (1, nx)
            D2 -> (2, ny, nx)
            D3 -> (3, nz, ny, nx)

    lattice : lattice class
        Must define:
            D
            Q
            c
            w
            cs2
            cs4
    """

    feq = np.empty((lattice.Q, *rho.shape))

    # |u|²
    u2 = np.sum(u**2, axis=0)

    inv_cs2 = 1.0 / lattice.cs2
    inv_2cs4 = 1.0 / (2 * lattice.cs4)
    inv_2cs2 = 1.0 / (2 * lattice.cs2)

    for i in range(lattice.Q):

        # e_i · u
        cu = np.zeros_like(rho)

        for d in range(lattice.D):
            cu += lattice.c[i, d] * u[d]

        feq[i] = (
            lattice.w[i]
            * rho
            * (
                1.0
                + inv_cs2 * cu
                + inv_2cs4 * cu**2
                - inv_2cs2 * u2
            )
        )

    return feq