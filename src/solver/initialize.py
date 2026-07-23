# solver/initialize.py

import numpy as np

from lattice.equilibrium import equilibrium


def initialize(nx, ny, rho0, ux0, uy0):

    rho = np.full((ny, nx), rho0)

    ux = np.full((ny, nx), ux0)

    uy = np.full((ny, nx), uy0)

    f = equilibrium(rho, ux, uy)

    return f, rho, ux, uy