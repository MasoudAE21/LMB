# boundary/bounce_back.py

from lattice.d2q9 import D2Q9


def apply_bounce_back(f):
    """
    Full-way bounce-back on left, right and bottom walls.
    Top wall is handled separately as a moving lid.
    """

    # ---------- Bottom ----------
    y = 0

    f[2, y] = f[4, y]
    f[5, y] = f[7, y]
    f[6, y] = f[8, y]

    # ---------- Top ----------
    # handled elsewhere

    # ---------- Left ----------
    x = 0

    f[1, :, x] = f[3, :, x]
    f[5, :, x] = f[7, :, x]
    f[8, :, x] = f[6, :, x]

    # ---------- Right ----------
    x = -1

    f[3, :, x] = f[1, :, x]
    f[6, :, x] = f[8, :, x]
    f[7, :, x] = f[5, :, x]

    return f