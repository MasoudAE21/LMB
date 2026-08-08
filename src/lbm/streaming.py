import numpy as np


def stream(f, lattice):

    streamed = np.empty_like(f)

    # -----------------------------
    # 1D
    # -----------------------------
    if lattice.D == 1:

        for i in range(lattice.Q):

            cx = lattice.c[i, 0]

            streamed[i] = np.roll(
                f[i],
                shift=cx
            )

            # Remove periodic wrapping
            if cx == 1:
                streamed[i, 0] = 0.0

            elif cx == -1:
                streamed[i, -1] = 0.0

    # -----------------------------
    # 2D
    # -----------------------------
    else:

        for i in range(lattice.Q):

            cx, cy = lattice.c[i]

            streamed[i] = np.roll(
                np.roll(
                    f[i],
                    shift=cy,
                    axis=0
                ),
                shift=cx,
                axis=1
            )

            # Remove y wrapping
            if cy == 1:
                streamed[i, 0, :] = 0.0

            elif cy == -1:
                streamed[i, -1, :] = 0.0

            # Remove x wrapping
            if cx == 1:
                streamed[i, :, 0] = 0.0

            elif cx == -1:
                streamed[i, :, -1] = 0.0

    return streamed