from lbm.equilibrium import (
    flow_equilibrium,
    scalar_equilibrium
)


def collide_flow(f, rho, u, tau, lattice):

    feq = flow_equilibrium(
        rho,
        u,
        lattice
    )

    return f - (f - feq) / tau


def collide_scalar(g, phi, tau, lattice, u=None):

    geq = scalar_equilibrium(
        phi,
        u,
        lattice
    )

    return g - (g - geq) / tau