import numpy as np
from lattice.d2q9 import D2Q9


def moving_lid(f, rho, U):

    y = -1

    # unknown populations after streaming
    unknown = [4, 7, 8]

    # opposite populations
    opposite = [2, 5, 6]

    uw = np.array([U, 0.0])

    for i, opp in zip(unknown, opposite):

        ei = D2Q9.c[i]

        correction = 6 * D2Q9.w[i] * rho[y] * np.dot(ei, uw)

        f[i, y] = f[opp, y] - correction

    return f