# solver/streaming.py

import numpy as np


def stream(populations, lattice):
    """
    Stream LBM populations without periodic wrapping.

    Parameters
    ----------
    populations : ndarray
        Distribution after collision.

        shape:
            (Q, nx)          for D1
            (Q, ny, nx)      for D2

    lattice : lattice class
        D1Q3, D2Q5, or D2Q9.

    Returns
    -------
    streamed : ndarray
        Streamed distribution.

    Notes
    -----
    Populations entering the computational domain
    from outside are left as zero.

    Boundary-condition routines must reconstruct
    those populations afterward.
    """

    populations = np.asarray(populations)

    expected_ndim = lattice.D + 1

    if populations.ndim != expected_ndim:
        raise ValueError(
            f"Expected populations with "
            f"{expected_ndim} dimensions, "
            f"got {populations.ndim}."
        )

    if populations.shape[0] != lattice.Q:
        raise ValueError(
            f"Expected Q={lattice.Q}, "
            f"got {populations.shape[0]} populations."
        )

    streamed = np.zeros_like(
        populations
    )

    for i in range(lattice.Q):

        velocity = lattice.c[i]

        source_slices = [
            slice(None)
            for _ in range(lattice.D)
        ]

        destination_slices = [
            slice(None)
            for _ in range(lattice.D)
        ]

        for d in range(lattice.D):

            shift = int(
                velocity[d]
            )

            # Physical dimensions:
            #
            # d = 0 -> x
            # d = 1 -> y
            #
            # Array dimensions:
            #
            # D1:
            #     (x,)
            #
            # D2:
            #     (y, x)
            #
            # Therefore physical and NumPy
            # axes are reversed.
            array_axis = (
                lattice.D - 1 - d
            )

            if shift > 0:

                source_slices[array_axis] = (
                    slice(0, -shift)
                )

                destination_slices[array_axis] = (
                    slice(shift, None)
                )

            elif shift < 0:

                source_slices[array_axis] = (
                    slice(-shift, None)
                )

                destination_slices[array_axis] = (
                    slice(0, shift)
                )

        source_index = (
            (i,)
            + tuple(source_slices)
        )

        destination_index = (
            (i,)
            + tuple(destination_slices)
        )

        streamed[destination_index] = (
            populations[source_index]
        )

    return streamed