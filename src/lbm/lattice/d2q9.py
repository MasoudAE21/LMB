# lattice/d2q9.py

import numpy as np


class D2Q9:

    # Discrete lattice velocities
    #
    #   6   2   5
    #     \ | /
    #   3---0---1
    #     / | \
    #   7   4   8
    #
    c = np.array([
        [ 0,  0],
        [ 1,  0],
        [ 0,  1],
        [-1,  0],
        [ 0, -1],
        [ 1,  1],
        [-1,  1],
        [-1, -1],
        [ 1, -1]
    ], dtype=np.int8)

    # Lattice weights
    w = np.array([
        4/9,
        1/9,
        1/9,
        1/9,
        1/9,
        1/36,
        1/36,
        1/36,
        1/36
    ], dtype=np.float64)

    # Opposite directions
    opp = np.array([
        0,
        3,
        4,
        1,
        2,
        7,
        8,
        5,
        6
    ], dtype=np.int8)

    Q = 9
    D = 2

    cs2 = 1.0 / 3.0
    cs4 = cs2**2