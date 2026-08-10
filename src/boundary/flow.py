def no_slip_walls(f, f_post, lattice):
    """
    Halfway bounce-back on all four stationary walls.
    """
    for i in range(lattice.Q):
        cx, cy = lattice.c[i]
        opposite = lattice.opp[i]
        # Left wall
        if cx < 0:
            f[opposite, :, 0] = (f_post[i, :, 0])
        # Right wall
        if cx > 0:
            f[opposite, :, -1] = (f_post[i, :, -1])
        # Bottom wall
        if cy < 0:
            f[opposite, 0, :] = (f_post[i, 0, :])
        # Top wall
        if cy > 0:
            f[opposite, -1, :] = (f_post[i, -1, :])
    return f    