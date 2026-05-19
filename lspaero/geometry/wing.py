"""Parametric flying-wing mesh generator.

Produces a thick-panel Mesh (upper surface, lower surface, tip cap) from
analytic wing geometry parameters.  The Mesh feeds directly into the solver
without modification.

Coordinate convention
---------------------
x : chordwise (LE → TE, positive downstream)
y : spanwise  (root → right tip, positive outboard)
z : normal    (positive up)

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §10.1.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .naca import naca4_camber, naca4_surfaces

# Integer labels stored in Mesh.surface_id
SURF_UPPER: int = 0
SURF_LOWER: int = 1
SURF_TIP: int = 2


def make_wing_mesh(
    half_span: float = 5.0,
    root_chord: float = 2.0,
    tip_chord: float = 1.0,
    sweep_le: float = 0.0,
    dihedral: float = 0.0,
    twist_tip: float = 0.0,
    airfoil: str = "0012",
    n_span: int = 20,
    n_chord: int = 10,
    include_tip_cap: bool = False,
) -> Mesh:
    """Build a right half-wing thick-panel Mesh.

    The mesh contains upper-surface and lower-surface panels, plus an optional
    tip-cap.  The trailing edge and root plane are left open for wake shedding
    and the symmetry image system respectively.

    Parameters
    ----------
    half_span : float
        Distance from root (y = 0) to wing tip (y = half_span).
    root_chord : float
        Chord length at the root (y = 0).
    tip_chord : float
        Chord length at the tip (y = half_span).
    sweep_le : float
        Leading-edge sweep angle in degrees (positive = swept back).
    dihedral : float
        Dihedral angle in degrees (positive = tip up).
    twist_tip : float
        Tip twist relative to root in degrees, linearly interpolated spanwise.
        Positive = nose up; washout is negative.
    airfoil : str
        NACA 4-digit code, e.g. ``'0012'`` or ``'2412'``.
    n_span : int
        Number of spanwise panels (cosine-clustered between root and tip).
    n_chord : int
        Number of chordwise panels per surface (cosine-clustered, LE → TE).
    include_tip_cap : bool
        If True, add ``n_chord`` tip-cap panels closing the wing at
        y = half_span.  Default False: leaving the tip open avoids the
        closed-body null-space of the vortex-ring AIC and keeps the linear
        system well-conditioned (κ ~ 10⁴ instead of ~ 10¹⁸).

    Returns
    -------
    Mesh
        Right half-wing mesh.  Panel count:
        ``n_chord * 2 * n_span`` (open tip) or
        ``n_chord * (2*n_span + 1)`` (with tip cap).
    """
    # ------------------------------------------------------------------ #
    # Spanwise layout — cosine clustering (dense at root and tip)         #
    # ------------------------------------------------------------------ #
    beta_s = np.linspace(0.0, np.pi, n_span + 1)
    eta = 0.5 * (1.0 - np.cos(beta_s))          # (n_span+1,)  [0, 1]

    y_sp = eta * half_span
    chord = root_chord + (tip_chord - root_chord) * eta
    x_le = y_sp * np.tan(np.radians(sweep_le))
    z_le = y_sp * np.tan(np.radians(dihedral))
    twist = np.radians(twist_tip) * eta           # (n_span+1,) radians

    # ------------------------------------------------------------------ #
    # Airfoil profile — unit chord, n_chord+1 nodes per surface (LE→TE)  #
    # ------------------------------------------------------------------ #
    xu, zu, xl, zl = naca4_surfaces(airfoil, n_chord)   # (n_chord+1,) each

    n_c = n_chord + 1    # chordwise nodes per surface
    n_s = n_span + 1     # spanwise nodes

    # ------------------------------------------------------------------ #
    # 3-D node grids — fully vectorised over chord × span                #
    #                                                                     #
    # Vertex layout: nodes[chordwise_i, spanwise_j], i in [0, n_chord],  #
    # j in [0, n_span].  Stored row-major → flat index = i*n_s + j.      #
    # ------------------------------------------------------------------ #
    xu2d = xu[:, None];  zu2d = zu[:, None]    # (n_c, 1)
    xl2d = xl[:, None];  zl2d = zl[:, None]

    cos_th = np.cos(twist)[None, :]            # (1, n_s)
    sin_th = np.sin(twist)[None, :]
    chord2d = chord[None, :]
    x_le2d  = x_le[None, :]
    z_le2d  = z_le[None, :]

    # Twist rotation about the LE in the local x-z plane:
    #   x_rot = x_unit * cos θ − z_unit * sin θ
    #   z_rot = x_unit * sin θ + z_unit * cos θ
    x_rot_u = xu2d * cos_th - zu2d * sin_th   # (n_c, n_s)
    z_rot_u = xu2d * sin_th + zu2d * cos_th
    x_rot_l = xl2d * cos_th - zl2d * sin_th
    z_rot_l = xl2d * sin_th + zl2d * cos_th

    y_grid = np.broadcast_to(y_sp[None, :], (n_c, n_s))  # (n_c, n_s)

    verts_upper = np.column_stack([
        (x_le2d + x_rot_u * chord2d).ravel(),
        y_grid.ravel(),
        (z_le2d + z_rot_u * chord2d).ravel(),
    ])   # (n_c * n_s, 3)

    verts_lower = np.column_stack([
        (x_le2d + x_rot_l * chord2d).ravel(),
        y_grid.ravel(),
        (z_le2d + z_rot_l * chord2d).ravel(),
    ])

    vertices = np.vstack([verts_upper, verts_lower])   # (2*n_c*n_s, 3)
    B = len(verts_upper)                                # = n_c * n_s

    # ------------------------------------------------------------------ #
    # Panel index arrays — vectorised meshgrid approach                   #
    #                                                                     #
    # Panel (i, j): chordwise strip i, spanwise strip j.                 #
    # Flat index in panels array = i * n_span + j.                       #
    # ------------------------------------------------------------------ #
    i_c = np.arange(n_chord)
    j_s = np.arange(n_span)
    II, JJ = np.meshgrid(i_c, j_s, indexing="ij")   # (n_chord, n_span)
    II = II.ravel()
    JJ = JJ.ravel()

    # Upper surface — CCW from above → normal points up (+z)
    # v0=fore-inner, v1=aft-inner, v2=aft-outer, v3=fore-outer
    pu0 =       II       * n_s +  JJ
    pu1 =      (II + 1)  * n_s +  JJ
    pu2 =      (II + 1)  * n_s + (JJ + 1)
    pu3 =       II       * n_s + (JJ + 1)
    panels_upper = np.column_stack([pu0, pu1, pu2, pu3])

    # Lower surface — CCW from below → normal points down (−z)
    # v0=fore-inner, v1=fore-outer, v2=aft-outer, v3=aft-inner
    pl0 = B +  II       * n_s +  JJ
    pl1 = B +  II       * n_s + (JJ + 1)
    pl2 = B + (II + 1)  * n_s + (JJ + 1)
    pl3 = B + (II + 1)  * n_s +  JJ
    panels_lower = np.column_stack([pl0, pl1, pl2, pl3])

    # Tip cap at j = n_span — normal points outboard (+y)
    # v0=upper-fore, v1=upper-aft, v2=lower-aft, v3=lower-fore
    ic = np.arange(n_chord)
    pt0 =     ic       * n_s + n_span
    pt1 =    (ic + 1)  * n_s + n_span
    pt2 = B + (ic + 1) * n_s + n_span
    pt3 = B +  ic      * n_s + n_span
    panels_tip = np.column_stack([pt0, pt1, pt2, pt3])

    if include_tip_cap:
        panels = np.vstack([panels_upper, panels_lower, panels_tip])
        n_up   = len(panels_upper)
        n_lo   = len(panels_lower)
        n_tip  = len(panels_tip)
        surface_id = np.concatenate([
            np.full(n_up,  SURF_UPPER, dtype=int),
            np.full(n_lo,  SURF_LOWER, dtype=int),
            np.full(n_tip, SURF_TIP,   dtype=int),
        ])
    else:
        panels = np.vstack([panels_upper, panels_lower])
        n_up   = len(panels_upper)
        n_lo   = len(panels_lower)
        surface_id = np.concatenate([
            np.full(n_up, SURF_UPPER, dtype=int),
            np.full(n_lo, SURF_LOWER, dtype=int),
        ])

    # ------------------------------------------------------------------ #
    # Trailing-edge pairs                                                 #
    # Last chordwise panel index for strip j:                            #
    #   upper: (n_chord − 1) * n_span + j                               #
    #   lower: n_up + (n_chord − 1) * n_span + j                        #
    # ------------------------------------------------------------------ #
    j_all = np.arange(n_span)
    te_upper = (n_chord - 1) * n_span + j_all
    te_lower = n_up + te_upper
    te_pairs = np.column_stack([te_upper, te_lower])   # (n_span, 2)

    # ------------------------------------------------------------------ #
    # Wake seed — midpoint of upper/lower TE edge per spanwise strip     #
    # ------------------------------------------------------------------ #
    # Upper TE node at spanwise index j: flat vertex index n_chord*n_s + j
    ute_j   = verts_upper[n_chord * n_s + j_all]       # (n_span, 3)
    ute_jp1 = verts_upper[n_chord * n_s + j_all + 1]
    lte_j   = verts_lower[n_chord * n_s + j_all]
    lte_jp1 = verts_lower[n_chord * n_s + j_all + 1]
    wake_seed = 0.25 * (ute_j + ute_jp1 + lte_j + lte_jp1)   # (n_span, 3)

    return Mesh(
        vertices=vertices,
        panels=panels,
        te_pairs=te_pairs,
        wake_seed=wake_seed,
        surface_id=surface_id,
    )


def make_vlm_mesh(
    half_span: float = 5.0,
    root_chord: float = 2.0,
    tip_chord: float = 1.0,
    sweep_le: float = 0.0,
    dihedral: float = 0.0,
    twist_tip: float = 0.0,
    airfoil: str = "0012",
    n_span: int = 20,
    n_chord: int = 10,
) -> Mesh:
    """Build a right half-wing VLM mesh on the mean camber surface.

    This mesh is the thin-surface input for the vortex-lattice solver.  Each
    panel has a single surface (``surface_id = SURF_UPPER = 0``).  The
    trailing-edge pairs store the same panel index in both columns because there
    is no thickness distinction on the camber surface.

    Parameters are identical to :func:`make_wing_mesh`.

    Returns
    -------
    Mesh
        Right half-wing camber-surface mesh with ``n_chord * n_span`` panels.
    """
    beta_s = np.linspace(0.0, np.pi, n_span + 1)
    eta = 0.5 * (1.0 - np.cos(beta_s))          # (n_span+1,)

    y_sp = eta * half_span
    chord = root_chord + (tip_chord - root_chord) * eta
    x_le = y_sp * np.tan(np.radians(sweep_le))
    z_le = y_sp * np.tan(np.radians(dihedral))
    twist = np.radians(twist_tip) * eta

    xc, zc = naca4_camber(airfoil, n_chord)      # (n_chord+1,) unit chord

    n_c = n_chord + 1    # chordwise nodes
    n_s = n_span + 1     # spanwise nodes

    xc2d = xc[:, None];  zc2d = zc[:, None]      # (n_c, 1)
    cos_th = np.cos(twist)[None, :]               # (1, n_s)
    sin_th = np.sin(twist)[None, :]
    chord2d = chord[None, :]
    x_le2d  = x_le[None, :]
    z_le2d  = z_le[None, :]

    x_rot = xc2d * cos_th - zc2d * sin_th        # (n_c, n_s)
    z_rot = xc2d * sin_th + zc2d * cos_th
    y_grid = np.broadcast_to(y_sp[None, :], (n_c, n_s))

    verts = np.column_stack([
        (x_le2d + x_rot * chord2d).ravel(),
        y_grid.ravel(),
        (z_le2d + z_rot * chord2d).ravel(),
    ])   # (n_c * n_s, 3)

    i_c = np.arange(n_chord)
    j_s = np.arange(n_span)
    II, JJ = np.meshgrid(i_c, j_s, indexing="ij")
    II = II.ravel(); JJ = JJ.ravel()

    # v0=fwd-inner, v1=aft-inner, v2=aft-outer, v3=fwd-outer
    # CCW from above → normal points +z
    p0 =  II       * n_s +  JJ
    p1 = (II + 1)  * n_s +  JJ
    p2 = (II + 1)  * n_s + (JJ + 1)
    p3 =  II       * n_s + (JJ + 1)
    panels = np.column_stack([p0, p1, p2, p3])   # (n_chord*n_span, 4)

    surface_id = np.zeros(len(panels), dtype=int)

    j_all = np.arange(n_span)
    te_idx = (n_chord - 1) * n_span + j_all      # last chordwise row
    te_pairs = np.column_stack([te_idx, te_idx])  # upper = lower (thin surface)

    te_inner = verts[n_chord * n_s + j_all]
    te_outer = verts[n_chord * n_s + j_all + 1]
    wake_seed = 0.5 * (te_inner + te_outer)

    return Mesh(
        vertices=verts,
        panels=panels,
        te_pairs=te_pairs,
        wake_seed=wake_seed,
        surface_id=surface_id,
    )
