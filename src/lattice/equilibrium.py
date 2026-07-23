# lattice/equilibrium.py

import numpy as np
from lattice.d2q9 import D2Q9


def equilibrium(rho, ux, uy):

    feq = np.zeros((9, *rho.shape))

    u2 = ux**2 + uy**2

    for i in range(D2Q9.Q):

        cu = (
            D2Q9.c[i,0]*ux +
            D2Q9.c[i,1]*uy
        )

        feq[i] = (
            D2Q9.w[i]
            * rho
            * (
                1
                + 3*cu
                + 4.5*cu**2
                - 1.5*u2
            )
        )

    return feq