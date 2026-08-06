# lattice/d1q3.py

import numpy as np


class D1Q3:

    # lattice velocities
    c = np.array([
        [ 0],
        [ 1],
        [ -1]
    ], dtype=np.int8)

    # lattice weights
    w = np.array([
        4/6,
        1/6,
        1/6
    ], dtype=np.float64)

    # opposite directions
    opp = np.array([
        0,
        2,
        1
    ], dtype=np.int8)

    Q = 3
    D = 1
    cs2 = 1 / 3