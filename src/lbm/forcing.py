import numpy as np


def buoyancy_force(rho, T, g_beta, T_ref):
    """
    Boussinesq buoyancy force
    force[0] = Fx = 0
    force[1] = Fy
    """

    force = np.zeros((2, *T.shape))
    force[1] = (rho * g_beta * (T - T_ref))
    return force


def get_source(force, lattice):
    source = np.empty((lattice.Q, *force.shape[1:]))
    Fx = force[0]
    Fy = force[1]
    cs2 = lattice.cs2
    for i in range(lattice.Q):
        cx, cy = lattice.c[i]
        # source[i] = (lattice.w[i] * factor * (cF / cs2 - uF / cs2 + cu * cF / cs4))
        source[i] = lattice.w[i] * Fy * cy / cs2
    return source