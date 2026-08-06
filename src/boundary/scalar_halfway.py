# boundary/scalar_halfway.py

import numpy as np


def _boundary_slice(
    lattice,
    axis,
    side
):
    """
    Construct NumPy slice for a planar boundary.

    Parameters
    ----------
    axis : int
        Physical axis.

        0 -> x
        1 -> y

    side : str
        "low" or "high"

        x-low  -> left
        x-high -> right

        y-low  -> bottom
        y-high -> top
    """

    if axis < 0 or axis >= lattice.D:
        raise ValueError(
            "Invalid physical axis."
        )

    if side not in ("low", "high"):
        raise ValueError(
            "side must be 'low' or 'high'."
        )

    array_axis = (
        lattice.D - 1 - axis
    )

    index = [
        slice(None)
        for _ in range(lattice.D)
    ]

    index[array_axis] = (
        0 if side == "low" else -1
    )

    return tuple(index)


def _outward_normal(
    lattice,
    axis,
    side
):
    """
    Outward normal vector for a planar wall.
    """

    normal = np.zeros(
        lattice.D,
        dtype=np.float64
    )

    normal[axis] = (
        -1.0
        if side == "low"
        else 1.0
    )

    return normal


def apply_dirichlet(
    g_streamed,
    g_post,
    lattice,
    axis,
    side,
    value
):
    """
    Apply stationary halfway bounce-back
    Dirichlet boundary condition.

    Implements:

        g_opp(i) =
            -g_i^+ + 2*w_i*phi_wall

    Parameters
    ----------
    g_streamed : ndarray
        Distribution after streaming.

    g_post : ndarray
        Post-collision distribution before streaming.

    lattice : lattice class
        D1Q3 or D2Q5.

    axis : int
        Physical normal axis:
            0 -> x
            1 -> y

    side : str
        "low" or "high"

    value : float or ndarray
        Prescribed wall scalar value.

    Returns
    -------
    g_streamed : ndarray
        Distribution with incoming boundary
        populations reconstructed.
    """

    boundary = _boundary_slice(
        lattice,
        axis,
        side
    )

    normal = _outward_normal(
        lattice,
        axis,
        side
    )

    for i in range(lattice.Q):

        ci = lattice.c[i]

        # Direction points from the fluid
        # node toward the wall/outside.
        if np.dot(ci, normal) <= 0:
            continue

        incoming = lattice.opp[i]

        out_index = (
            (i,)
            + boundary
        )

        in_index = (
            (incoming,)
            + boundary
        )

        g_streamed[in_index] = (
            -g_post[out_index]
            + 2.0
            * lattice.w[i]
            * value
        )

    return g_streamed


def apply_general(
    g_streamed,
    g_post,
    phi,
    lattice,
    axis,
    side,
    b1,
    b2,
    b3,
    dx=1.0
):
    """
    Apply the general halfway scalar boundary:

        b1 * d(phi)/dn
        + b2 * phi
        = b3

    This implements the wall-value calculation
    corresponding to Eq. (15) of Zhang et al.,
    followed by the halfway bounce-back rule.

    Parameters
    ----------
    g_streamed : ndarray
        Distribution after streaming.

    g_post : ndarray
        Distribution after collision,
        before streaming.

    phi : ndarray
        Current macroscopic scalar field.

    lattice : lattice class
        D1Q3 or D2Q5.

    axis : int
        Physical boundary axis.

    side : str
        "low" or "high".

    b1, b2, b3 : float
        Boundary coefficients.

    dx : float
        Lattice spacing.

    Returns
    -------
    g_streamed : ndarray
    """

    # Dirichlet:
    #
    # b2 * phi_wall = b3
    if np.isclose(b1, 0.0):

        if np.isclose(b2, 0.0):
            raise ValueError(
                "Invalid boundary condition: "
                "b1 and b2 cannot both be zero."
            )

        wall_value = (
            b3 / b2
        )

        return apply_dirichlet(
            g_streamed=g_streamed,
            g_post=g_post,
            lattice=lattice,
            axis=axis,
            side=side,
            value=wall_value
        )

    boundary = _boundary_slice(
        lattice,
        axis,
        side
    )

    normal = _outward_normal(
        lattice,
        axis,
        side
    )

    phi_f = phi[boundary]

    for i in range(lattice.Q):

        ci = lattice.c[i]

        n_dot_ci = np.dot(
            normal,
            ci
        )

        if n_dot_ci <= 0:
            continue

        # Eq. (15):
        #
        # phi_w =
        # [phi_f + 0.5 dx (n.c_i) b3/b1]
        # --------------------------------
        # [1 + 0.5 dx (n.c_i) b2/b1]

        numerator = (
            phi_f
            + 0.5
            * dx
            * n_dot_ci
            * b3
            / b1
        )

        denominator = (
            1.0
            + 0.5
            * dx
            * n_dot_ci
            * b2
            / b1
        )

        wall_value = (
            numerator
            / denominator
        )

        incoming = lattice.opp[i]

        out_index = (
            (i,)
            + boundary
        )

        in_index = (
            (incoming,)
            + boundary
        )

        g_streamed[in_index] = (
            -g_post[out_index]
            + 2.0
            * lattice.w[i]
            * wall_value
        )

    return g_streamed


def apply_neumann(
    g_streamed,
    g_post,
    phi,
    lattice,
    axis,
    side,
    gradient,
    dx=1.0
):
    """
    Convenience wrapper for:

        d(phi)/dn = gradient

    Corresponds to:

        b1 = 1
        b2 = 0
        b3 = gradient

    For an adiabatic wall:

        gradient = 0
    """

    return apply_general(
        g_streamed=g_streamed,
        g_post=g_post,
        phi=phi,
        lattice=lattice,
        axis=axis,
        side=side,
        b1=1.0,
        b2=0.0,
        b3=gradient,
        dx=dx
    )