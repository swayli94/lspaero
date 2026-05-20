"""Parametric axisymmetric fuselage mesh generator.

Generates a closed surface mesh for a body of revolution (fuselage), which
can be combined with a wing mesh via ``combine_meshes`` for wing-body panel
method analysis.

All panels carry ``lifting_panels = False``: they receive source-only
strengths with no circulation, Kutta condition, or wake shedding.

Coordinate convention: x chordwise (nose → tail), y spanwise (right), z up.
The body axis is the x-axis (y = 0, z = 0).

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §4.1–4.2.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh

SURF_BODY: int = 3
"""``surface_id`` label for fuselage panels (0=upper, 1=lower, 2=tip, 3=body)."""


def sears_haack_profile(length: float, r_max: float):
    """Sears–Haack body radius profile r(x).

    The Sears–Haack body minimises wave drag at a given volume.  Its profile
    is the classical solution to the area-rule optimisation:

        r(x) = r_max · (1 − ξ²)^(3/4),   ξ = 2x/L − 1 ∈ [−1, 1]

    At x = 0 and x = L the radius is zero (pointed nose and tail).

    Parameters
    ----------
    length : float
        Total body length L.
    r_max : float
        Maximum cross-sectional radius, attained at x = L/2.

    Returns
    -------
    r : callable
        r(x) for x in [0, length].  Accepts scalar or array input.

    Reference
    ---------
    Sears, W. R. (1947), "On Projectiles of Minimum Wave Drag", *Q. Appl. Math.*
    5(4): 361–366.  Katz & Plotkin §15.3.
    """
    def r(x):
        xi = 2.0 * np.asarray(x, dtype=float) / length - 1.0
        xi = np.clip(xi, -1.0, 1.0)
        return r_max * (1.0 - xi ** 2) ** 0.75

    return r


def make_body_mesh(
    length: float = 10.0,
    r_profile=None,
    n_axial: int = 20,
    n_circ: int = 16,
    x_offset: float = 0.0,
) -> Mesh:
    """Build a full-360° axisymmetric fuselage mesh.

    The fuselage is modelled as a complete, closed body of revolution about
    the x-axis.  The mesh contains:

    * **n_circ nose-cap triangular panels** (degenerate quads) connecting the
      nose apex to the first circumferential ring.
    * **(n_axial − 1) × n_circ cylindrical-body quad panels** connecting
      adjacent ring pairs.
    * **n_circ tail-cap triangular panels** connecting the last ring to the
      tail apex.

    Total panels: ``n_circ × (n_axial + 1)``.

    All panels have ``lifting_panels = False`` and ``surface_id = SURF_BODY``
    (= 3).  ``te_pairs`` and ``wake_seed`` are empty.

    The full 360° model is required because the solver's y → −y image system
    is only applied to *lifting* panels (the wing).  Non-lifting body panels
    are fully explicit and do not use the image system.

    Parameters
    ----------
    length : float
        Fuselage length (nose apex to tail apex).
    r_profile : callable
        Cross-sectional radius as a function of axial coordinate:
        ``r = r_profile(x)`` for ``x ∈ [0, length]``.  Should satisfy
        ``r_profile(0) ≈ 0`` and ``r_profile(length) ≈ 0`` for a closed
        body.  Use :func:`sears_haack_profile` for the Sears–Haack body.
    n_axial : int
        Number of intermediate circumferential rings (where ``r > 0``).
        Cosine-clustered between nose and tail for better tip resolution.
    n_circ : int
        Number of circumferential panels per ring (panels around the body).
        Must be even for y-symmetric visualisation.
    x_offset : float
        x-coordinate of the nose apex.  Allows positioning the fuselage
        relative to the wing leading edge.

    Returns
    -------
    Mesh
        Full-360° closed fuselage mesh with ``n_circ × (n_axial + 1)`` panels
        and ``lifting_panels = False`` for all panels.

    Notes
    -----
    Vertex coordinate convention:
        ``y = r · cos(θ)``,  ``z = r · sin(θ)``  with θ ∈ [0, 2π).

    Panel normal convention:
        Computed via the diagonal cross-product → points radially outward
        (away from the x-axis) on cylindrical panels; forward-outward on the
        nose cap; aft-outward on the tail cap.
    """
    if r_profile is None:
        raise ValueError(
            "r_profile must be provided.  "
            "Use sears_haack_profile(length, r_max) for the Sears–Haack body."
        )

    # ------------------------------------------------------------------ #
    # Axial stations for intermediate rings (cosine clustering)           #
    # ------------------------------------------------------------------ #
    # n_axial inner rings at t = pi*i/(n_axial+1) for i = 1..n_axial
    t_inner = np.linspace(0.0, np.pi, n_axial + 2)[1:-1]   # (n_axial,)
    xi = 0.5 * (1.0 - np.cos(t_inner))                      # cosine in [0, 1]
    x_rings = x_offset + xi * length                         # (n_axial,) ring x-coords

    r_rings = r_profile(x_rings - x_offset)                  # (n_axial,) ring radii
    r_rings = np.maximum(r_rings, 0.0)                        # safety: no negative r

    # ------------------------------------------------------------------ #
    # Circumferential angles                                               #
    # y = r·cos(θ),  z = r·sin(θ),  θ ∈ [0, 2π)                        #
    # ------------------------------------------------------------------ #
    theta = 2.0 * np.pi * np.arange(n_circ) / n_circ         # (n_circ,)

    # ------------------------------------------------------------------ #
    # Vertices                                                             #
    # Layout:                                                             #
    #   index 0             : nose apex (x_offset, 0, 0)                 #
    #   index 1..n_axial*n_circ : rings [i=0..n_axial-1, j=0..n_circ-1] #
    #     flat index = 1 + i*n_circ + j                                  #
    #   index n_axial*n_circ+1 : tail apex (x_offset+length, 0, 0)      #
    # ------------------------------------------------------------------ #
    nose_apex = np.array([[x_offset, 0.0, 0.0]])
    tail_apex = np.array([[x_offset + length, 0.0, 0.0]])

    # Ring vertices: (n_axial, n_circ, 3) → flatten to (n_axial*n_circ, 3)
    x_grid = np.repeat(x_rings, n_circ)               # (n_axial*n_circ,)
    y_grid = np.outer(r_rings, np.cos(theta)).ravel()  # (n_axial*n_circ,)
    z_grid = np.outer(r_rings, np.sin(theta)).ravel()  # (n_axial*n_circ,)
    ring_verts = np.column_stack([x_grid, y_grid, z_grid])

    vertices = np.vstack([nose_apex, ring_verts, tail_apex])
    # Total: 1 + n_axial*n_circ + 1 = n_axial*n_circ + 2

    nose_apex_idx = 0
    tail_apex_idx = 1 + n_axial * n_circ

    def _ring_idx(i: int, j: int) -> int:
        """Flat vertex index for ring i (0-based), circumferential position j."""
        return 1 + i * n_circ + (j % n_circ)

    # ------------------------------------------------------------------ #
    # Panels                                                               #
    # ------------------------------------------------------------------ #
    panels_list = []

    # -- Nose cap: n_circ triangles (degenerate quads with v2 = v3) ------ #
    # Winding gives outward normal pointing forward-outward (negative x,   #
    # positive radial) — verified analytically in ROADMAP Stage 9.        #
    # Panel: [apex, ring[0,(j+1)%nc], ring[0,j], ring[0,j]]              #
    for j in range(n_circ):
        p = [
            nose_apex_idx,
            _ring_idx(0, j + 1),
            _ring_idx(0, j),
            _ring_idx(0, j),   # degenerate: v2 = v3
        ]
        panels_list.append(p)

    # -- Cylindrical body: (n_axial-1)*n_circ quads ---------------------- #
    # Panel connecting ring i and ring i+1:                                #
    #   v0 = ring[i, j],    v1 = ring[i, j+1]                            #
    #   v2 = ring[i+1,j+1], v3 = ring[i+1,j]                            #
    # This winding gives a radially outward normal (verified analytically).#
    for i in range(n_axial - 1):
        for j in range(n_circ):
            p = [
                _ring_idx(i,     j),
                _ring_idx(i,     j + 1),
                _ring_idx(i + 1, j + 1),
                _ring_idx(i + 1, j),
            ]
            panels_list.append(p)

    # -- Tail cap: n_circ triangles (degenerate quads with v2 = v3) ------ #
    # Winding gives outward normal pointing aft-outward (positive x,       #
    # positive radial) — verified analytically.                           #
    # Panel: [tail_apex, ring[-1,j], ring[-1,(j+1)%nc], ring[-1,(j+1)%nc]]#
    for j in range(n_circ):
        p = [
            tail_apex_idx,
            _ring_idx(n_axial - 1, j),
            _ring_idx(n_axial - 1, j + 1),
            _ring_idx(n_axial - 1, j + 1),   # degenerate: v2 = v3
        ]
        panels_list.append(p)

    panels = np.array(panels_list, dtype=int)
    n_panels = len(panels)   # = n_circ*(n_axial+1)

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #
    surface_id      = np.full(n_panels, SURF_BODY, dtype=int)
    lifting_panels  = np.zeros(n_panels, dtype=bool)          # non-lifting
    te_pairs        = np.empty((0, 2), dtype=int)
    wake_seed       = np.empty((0, 3), dtype=float)

    return Mesh(
        vertices=vertices,
        panels=panels,
        te_pairs=te_pairs,
        wake_seed=wake_seed,
        surface_id=surface_id,
        lifting_panels=lifting_panels,
    )
