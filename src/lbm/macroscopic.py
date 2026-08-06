# solver/macroscopic.py

import numpy as np
from lattice.d2q9 import D2Q9


def macroscopic(f):

    rho = np.sum(f, axis=0)

    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)

    for i in range(9):

        ux += D2Q9.c[i,0] * f[i]
        uy += D2Q9.c[i,1] * f[i]

    ux /= rho
    uy /= rho

    return rho, ux, uy