import numpy as np


def scalar_macroscopic(g):
    """
    Recover temperature/scalar field.
    """
    return np.sum(g, axis=0)


def flow_macroscopic(f, lattice, force=None):
    """
    Recover density and velocity
    f : (Q, ny, nx)
    returns
    -------
    rho : (ny, nx)
    u   : (2, ny, nx)
    """
    
    rho = np.sum(f, axis=0)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    
    for i in range(lattice.Q):
        cx, cy = lattice.c[i]
        ux += cx * f[i]
        uy += cy * f[i]

    # Later needed for buoyancy in Problem 1
    if force is not None:
        ux += 0.5 * force[0]
        uy += 0.5 * force[1]

    ux /= rho
    uy /= rho
    u = np.array([ux, uy])
    return rho, u