def dirichlet_left(g, g_post, T_wall, lattice):

    for i in range(lattice.Q):

        # Population moving out through left wall
        if lattice.c[i, 0] < 0:

            opposite = lattice.opp[i]

            if lattice.D == 1:

                g[opposite, 0] = (
                    -g_post[i, 0]
                    + 2.0
                    * lattice.w[i]
                    * T_wall
                )

            else:

                g[opposite, :, 0] = (
                    -g_post[i, :, 0]
                    + 2.0
                    * lattice.w[i]
                    * T_wall
                )

    return g


def dirichlet_right(g, g_post, T_wall, lattice):

    for i in range(lattice.Q):

        # Population moving out through right wall
        if lattice.c[i, 0] > 0:

            opposite = lattice.opp[i]

            if lattice.D == 1:

                g[opposite, -1] = (
                    -g_post[i, -1]
                    + 2.0
                    * lattice.w[i]
                    * T_wall
                )

            else:

                g[opposite, :, -1] = (
                    -g_post[i, :, -1]
                    + 2.0
                    * lattice.w[i]
                    * T_wall
                )

    return g


def adiabatic_bottom(g, g_post, T, lattice):

    for i in range(lattice.Q):

        # Population moving out through bottom wall
        if lattice.c[i, 1] < 0:

            opposite = lattice.opp[i]

            g[opposite, 0, 1:-1] = (
                -g_post[i, 0, 1:-1]
                + 2.0
                * lattice.w[i]
                * T[0, 1:-1]
            )

    return g


def adiabatic_top(g, g_post, T, lattice):

    for i in range(lattice.Q):

        # Population moving out through top wall
        if lattice.c[i, 1] > 0:

            opposite = lattice.opp[i]

            g[opposite, -1, 1:-1] = (
                -g_post[i, -1, 1:-1]
                + 2.0
                * lattice.w[i]
                * T[-1, 1:-1]
            )

    return g