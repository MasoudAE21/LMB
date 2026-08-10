import numpy as np


def flow_equilibrium(rho, u, lattice):
    """
    D2Q9 equilibrium distribution for fluid flow.

    rho : (ny, nx)
    u   : (2, ny, nx)
    """

    feq = np.empty((lattice.Q, *rho.shape))
    ux = u[0]
    uy = u[1]
    u2 = ux**2 + uy**2
    cs2 = lattice.cs2
    cs4 = cs2**2

    for i in range(lattice.Q):
        cx, cy = lattice.c[i]
        cu = cx * ux + cy * uy
        feq[i] = (lattice.w[i] * rho * (1.0 + cu / cs2 + cu**2 / (2.0 * cs4) - u2 / (2.0 * cs2)))

    return feq


def scalar_equilibrium(phi, u, lattice):
    """
    Equilibrium distribution for temperature/scalar transport
    D1Q3:
        phi : (nx,)
        u   : None or (1, nx)
    D2Q5:
        phi : (ny, nx)
        u   : (2, ny, nx)
    """
    
    geq = np.empty((lattice.Q, *phi.shape))
    # Pure diffusion
    if u is None:
        for i in range(lattice.Q):
            geq[i] = lattice.w[i] * phi
        return geq

    # Convection-diffusion
    for i in range(lattice.Q):
        if lattice.D == 1:
            cu = lattice.c[i, 0] * u[0]
        else:
            cx, cy = lattice.c[i]
            cu = (cx * u[0] + cy * u[1])
        geq[i] = (lattice.w[i] * phi * (1.0 + cu / lattice.cs2))
    return geq