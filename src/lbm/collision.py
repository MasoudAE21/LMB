from lbm.equilibrium import (flow_equilibrium, scalar_equilibrium)


def collide_flow(f, rho, u, tau, lattice, source=None):

    feq = flow_equilibrium(rho, u, lattice)
    f_post = f - (f - feq) / tau
    if source is not None:
        f_post += source
    
    return f_post


def collide_scalar(g, phi, tau, lattice, u=None):

    geq = scalar_equilibrium(phi, u, lattice)

    return g - (g - geq) / tau