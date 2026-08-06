# lattice/d2q9.py

import numpy as np


class D2Q5:

    # lattice velocities
    c = np.array([
        [ 0, 0],
        [ 1, 0],
        [ 0, 1],
        [-1, 0],
        [ 0,-1]
    ], dtype=np.int8)

    # lattice weights
    w = np.array([
        2/6,
        1/6,
        1/6,
        1/6,
        1/6
    ], dtype=np.float64)

    # opposite directions
    opp = np.array([
        0,
        3,
        4,
        1,
        2
    ], dtype=np.int8)

    Q = 5
    D = 2
    cs2 = 1 / 3