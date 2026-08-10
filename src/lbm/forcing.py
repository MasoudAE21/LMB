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


def guo_source(force, u, tau, lattice):
    """
    Guo forcing term for D2Q9.
    """

    source = np.empty((lattice.Q, *force.shape[1:]))
    ux = u[0]
    uy = u[1]
    Fx = force[0]
    Fy = force[1]
    uF = (ux * Fx + uy * Fy)
    cs2 = lattice.cs2
    cs4 = cs2**2
    factor = (1.0 - 1.0 / (2.0 * tau))
    for i in range(lattice.Q):
        cx, cy = lattice.c[i]
        cu = (cx * ux + cy * uy)
        cF = (cx * Fx + cy * Fy)
        source[i] = (lattice.w[i] * factor * (cF / cs2 - uF / cs2 + cu * cF / cs4))
    return source