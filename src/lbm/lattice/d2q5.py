# lattice/d2q5.py

import numpy as np


class D2Q5:
    """D2Q5 lattice for two-dimensional scalar transport."""

    # Discrete lattice velocities
    #
    #       2
    #       |
    #   3---0---1
    #       |
    #       4
    #
    c = np.array([
        [ 0,  0],
        [ 1,  0],
        [ 0,  1],
        [-1,  0],
        [ 0, -1]
    ], dtype=np.int8)

    # Lattice weights
    w = np.array([
        2/6,
        1/6,
        1/6,
        1/6,
        1/6
    ], dtype=np.float64)

    # Opposite directions
    opp = np.array([
        0,
        3,
        4,
        1,
        2
    ], dtype=np.int8)

    Q = 5
    D = 2

    cs2 = 1.0 / 3.0
    cs4 = cs2**2