# lattice/d1q3.py

import numpy as np


class D1Q3:
    """D1Q3 lattice for one-dimensional scalar diffusion."""

    # Discrete lattice velocities
    c = np.array([
        [0],
        [1],
        [-1]
    ], dtype=np.int8)

    # Lattice weights
    w = np.array([
        4/6,
        1/6,
        1/6
    ], dtype=np.float64)

    # Opposite directions
    opp = np.array([
        0,
        2,
        1
    ], dtype=np.int8)

    Q = 3
    D = 1

    cs2 = 1.0 / 3.0
    cs4 = cs2**2