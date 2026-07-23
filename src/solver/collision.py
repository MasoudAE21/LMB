# solver/collision.py

from lattice.equilibrium import equilibrium


def collide(f, rho, ux, uy, tau):

    feq = equilibrium(rho, ux, uy)

    f += -(f - feq)/tau

    return f