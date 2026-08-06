# solver/collision.py

import numpy as np

from lbm.equilibrium import (
    flow_equilibrium,
    scalar_equilibrium,
)


def bgk_collision(
    populations,
    equilibrium,
    tau,
    source=None
):
    """
    Generic BGK collision step.

    f_post = f - (f - f_eq) / tau

    Parameters
    ----------
    populations : ndarray
        Current distribution populations.

    equilibrium : ndarray
        Equilibrium populations.

    tau : float
        Relaxation time.

    source : ndarray or None
        Optional source/forcing term.

        This will later be useful for the
        buoyancy force in Problem 1.

    Returns
    -------
    post_collision : ndarray
        Distribution after collision.
    """

    if tau <= 0.5:
        raise ValueError(
            "tau must be greater than 0.5 "
            "for positive transport coefficients."
        )

    post_collision = (
        populations
        - (populations - equilibrium) / tau
    )

    if source is not None:
        post_collision = (
            post_collision + source
        )

    return post_collision


def collide_flow(
    f,
    rho,
    u,
    tau,
    lattice,
    source=None
):
    """
    BGK collision for hydrodynamic populations.
    """

    feq = flow_equilibrium(
        rho=rho,
        u=u,
        lattice=lattice
    )

    return bgk_collision(
        populations=f,
        equilibrium=feq,
        tau=tau,
        source=source
    )


def collide_scalar(
    g,
    phi,
    tau,
    lattice,
    u=None
):
    """
    BGK collision for scalar populations.
    """

    geq = scalar_equilibrium(
        phi=phi,
        u=u,
        lattice=lattice
    )

    return bgk_collision(
        populations=g,
        equilibrium=geq,
        tau=tau
    )