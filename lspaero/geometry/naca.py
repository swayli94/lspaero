"""NACA 4-digit airfoil generator with cosine-clustered spacing.

Reference: Abbott & von Doenhoff, "Theory of Wing Sections", Dover (1959),
           Chapter 6.
"""
import numpy as np


def naca4(code: str, n_panels: int, closed_te: bool = True):
    """Generate NACA 4-digit airfoil panel nodes with cosine spacing.

    Nodes are arranged counter-clockwise (CCW) as seen from outside the body:
    lower trailing edge → leading edge → upper trailing edge.  This ordering
    makes the panel left-hand normal point outward on both surfaces.

    Parameters
    ----------
    code : str
        NACA 4-digit designation, e.g. ``'0012'``, ``'2412'``.
    n_panels : int
        Total number of panels (must be even; split equally over upper and
        lower surfaces).
    closed_te : bool
        If True use the modified last coefficient (−0.1036) that closes the
        trailing edge exactly.  If False use −0.1015 (standard open-TE form).

    Returns
    -------
    x, z : ndarray, shape (n_panels + 1,)
        Panel node coordinates.  Unit chord (c = 1).

    Raises
    ------
    ValueError
        If *n_panels* is odd or *code* is not a 4-digit NACA designator.
    """
    code = str(code).strip().zfill(4)
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Expected a 4-digit NACA code, got '{code}'")
    if n_panels % 2 != 0:
        raise ValueError("n_panels must be even")

    m = int(code[0]) / 100.0   # maximum camber fraction of chord
    p = int(code[1]) / 10.0    # chordwise position of max camber
    t = int(code[2:4]) / 100.0  # maximum thickness fraction of chord

    n_half = n_panels // 2
    beta = np.linspace(0.0, np.pi, n_half + 1)
    xc = 0.5 * (1.0 - np.cos(beta))  # cosine clustering, 0 → 1

    # Thickness distribution (Abbott & von Doenhoff Eq. 6.1)
    a4 = -0.1036 if closed_te else -0.1015
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        + a4   * xc**4
    )

    # Mean camber line and its slope
    if m < 1e-10 or p < 1e-10:
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
    else:
        fwd = xc <= p
        yc = np.where(
            fwd,
            (m / p**2) * (2.0 * p * xc - xc**2),
            (m / (1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc**2),
        )
        dyc_dx = np.where(
            fwd,
            (2.0 * m / p**2) * (p - xc),
            (2.0 * m / (1.0 - p)**2) * (p - xc),
        )

    theta = np.arctan(dyc_dx)
    xu = xc - yt * np.sin(theta)
    zu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    zl = yc - yt * np.cos(theta)

    # CCW arrangement: lower TE → LE → upper TE
    x = np.concatenate([xl[::-1], xu[1:]])
    z = np.concatenate([zl[::-1], zu[1:]])
    return x, z
