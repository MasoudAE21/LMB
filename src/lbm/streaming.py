# solver/streaming.py

import numpy as np
from lattice.d2q9 import D2Q9


def stream(f):

    streamed = np.empty_like(f)

    for i in range(9):

        cx, cy = D2Q9.c[i]

        streamed[i] = np.roll(
            np.roll(
                f[i],
                shift=cy,
                axis=0
            ),
            shift=cx,
            axis=1
        )

    return streamed