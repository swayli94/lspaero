"""Mesh topology utilities for triangulated surface meshes (OpenVSP / Cart3D).

Provides trailing-edge detection and ``Mesh`` assembly from ``.tri`` data.

Key algorithm
-------------
For OpenVSP wing meshes each *lifting-surface component* is a closed thick
surface (upper + lower + LE + TE + tip-cap stitched together).  The trailing
edge is the interior crease between the upper and lower surface panels; it
appears as a dihedral angle of ≈ 166° between adjacent panels.  Wing-body
junction edges have a smaller crease (≈ 122°) and are excluded by setting
the detection threshold to 150°.

Triangles from OpenVSP are stored as *degenerate quads* in ``Mesh.panels``
by repeating the last vertex: ``(i, j, k) → panels row (i, j, k, k)``.
This is transparent to the solver:

* ``_compute_derived``: the diagonal cross-product gives the correct area and
  normal for both quads and degenerate quads.
* ``build_panel_aic``: the zero-length segment ``v[k]→v[k]`` contributes
  nothing to the Biot-Savart sum.
* Wake geometry in ``build_panel_aic``: uses ``mesh.te_verts`` (added in
  this stage) instead of inferring endpoints from panel vertex ordering.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §10.1.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .mesh import Mesh


# ---------------------------------------------------------------------------
# Low-level edge topology
# ---------------------------------------------------------------------------

def build_edge_table(
    triangles: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """Build an undirected edge → triangle-index mapping.

    Parameters
    ----------
    triangles : (Nt, 3) int   0-based vertex indices.

    Returns
    -------
    edge_table : dict  (v_a, v_b) with v_a < v_b  →  [tri_idx, ...]
        Each key is a sorted vertex pair; each value is a list of the 1 or 2
        triangle indices that share that edge.
    """
    edge_table: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti, tri in enumerate(triangles):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (int(min(tri[a], tri[b])), int(max(tri[a], tri[b])))
            edge_table[key].append(ti)
    return dict(edge_table)


# ---------------------------------------------------------------------------
# TE detection
# ---------------------------------------------------------------------------

def detect_te_edges(
    vertices: np.ndarray,
    triangles: np.ndarray,
    comp_ids: np.ndarray,
    lifting_comp_ids: list[int],
    te_angle_thresh: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect trailing-edge panel pairs in a triangulated lifting surface.

    The trailing edge is identified as interior edges *within a single
    lifting-surface component* whose adjacent triangles form a crease angle
    greater than ``te_angle_thresh``.  Cross-component edges (wing-body
    junction) and smooth-surface edges are excluded.

    Upper vs. lower panel distinction: the triangle with the higher mean-z
    centroid at the TE edge is labelled *upper*; the other is *lower*.
    For a horizontally-flying wing this is unambiguous.

    The results are sorted in ascending spanwise order (by the y-midpoint of
    each TE edge) so that ``te_pairs`` rows correspond to root-to-tip strips.

    Parameters
    ----------
    vertices : (Nv, 3) float
    triangles : (Nt, 3) int  (0-based)
    comp_ids : (Nt,) int
    lifting_comp_ids : list[int]
        Component IDs that carry lift (Kutta condition + wake).  Non-lifting
        body components are excluded from TE detection.
    te_angle_thresh : float
        Minimum dihedral angle (degrees) between adjacent triangle normals to
        qualify as a trailing edge.  Default 150° discriminates OpenVSP TE
        (≈ 166°) from wing-body junctions (≈ 122°).

    Returns
    -------
    te_pairs : (Nte, 2) int
        ``[upper_tri_idx, lower_tri_idx]`` for each TE edge, sorted spanwise.
    te_verts : (Nte, 2, 3) float
        ``[[inner_TE_point, outer_TE_point], ...]`` — the two endpoints of
        each TE edge, sorted so index 0 is closer to the root (smaller |y|).
    wake_seed : (Nte, 3) float
        Midpoint of each TE edge — starting point for wake filaments.
    """
    cos_thresh = np.cos(np.radians(te_angle_thresh))
    lifting_set = set(lifting_comp_ids)

    # Triangle normals (unnormalized, then normalised)
    v = vertices[triangles]                         # (Nt, 3, 3)
    e1 = v[:, 1, :] - v[:, 0, :]
    e2 = v[:, 2, :] - v[:, 0, :]
    raw_normals = np.cross(e1, e2)                  # (Nt, 3)
    mag = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    normals = raw_normals / np.where(mag > 0, mag, 1.0)

    # Triangle centroids
    centroids = v.mean(axis=1)                      # (Nt, 3)

    edge_table = build_edge_table(triangles)

    te_rows: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    # (y_mid, upper_ti, lower_ti, inner_pt, outer_pt)

    for (va, vb), tris in edge_table.items():
        if len(tris) != 2:
            continue                                # boundary or non-manifold

        ti, tj = tris

        # Both triangles must belong to lifting components
        if comp_ids[ti] not in lifting_set or comp_ids[tj] not in lifting_set:
            continue

        # Both must belong to the *same* component (exclude cross-junction edges)
        if comp_ids[ti] != comp_ids[tj]:
            continue

        # Crease-angle criterion
        cos_a = float(np.dot(normals[ti], normals[tj]))
        if cos_a > cos_thresh:                      # angle ≤ threshold → smooth
            continue

        # Identify upper / lower by centroid z
        if centroids[ti, 2] >= centroids[tj, 2]:
            upper_ti, lower_ti = ti, tj
        else:
            upper_ti, lower_ti = tj, ti

        # TE edge endpoints: sort by |y| so index 0 = inner (root-side)
        pt_a = vertices[va]
        pt_b = vertices[vb]
        if abs(pt_a[1]) <= abs(pt_b[1]):
            inner_pt, outer_pt = pt_a, pt_b
        else:
            inner_pt, outer_pt = pt_b, pt_a

        y_mid = 0.5 * (pt_a[1] + pt_b[1])
        te_rows.append((y_mid, upper_ti, lower_ti, inner_pt, outer_pt))

    if not te_rows:
        raise RuntimeError(
            "No trailing-edge edges detected.  Check lifting_comp_ids and "
            "te_angle_thresh."
        )

    # Sort spanwise (ascending y_mid → root to tip)
    te_rows.sort(key=lambda r: r[0])

    Nte = len(te_rows)
    te_pairs   = np.empty((Nte, 2), dtype=int)
    te_verts   = np.empty((Nte, 2, 3), dtype=float)
    wake_seed  = np.empty((Nte, 3), dtype=float)

    for k, (y_mid, up, lo, inn, out) in enumerate(te_rows):
        te_pairs[k]      = [up, lo]
        te_verts[k, 0]   = inn
        te_verts[k, 1]   = out
        wake_seed[k]     = 0.5 * (inn + out)

    return te_pairs, te_verts, wake_seed


# ---------------------------------------------------------------------------
# Quad → triangle conversion
# ---------------------------------------------------------------------------

def triangulate_mesh(mesh: Mesh) -> Mesh:
    """Convert each quad panel to a degenerate-quad (triangular) panel.

    Each quad ``(v0, v1, v2, v3)`` is replaced by the **first-diagonal**
    degenerate quad ``(v0, v1, v2, v2)``.  The panel count is unchanged;
    only the 4th vertex index is replaced with v2 (same index repeated).

    This representation is compatible with every solver:

    * ``Mesh._compute_derived``: the centroid uses only the 3 unique vertices;
      area and normal are exact.
    * ``_ring_points`` (VLM): detects the degenerate quad and recovers the
      missing fwd-outer vertex via parallelogram completion
      ``v3_virtual = v0 + (v2 − v1)``.  For rectangular panels this is exact;
      for swept/tapered panels the error is O(taper / sweep).
    * ``build_panel_aic`` (vortex ring): the zero-length segment v2→v2
      contributes nothing to the Biot-Savart sum.
    * ``build_source_aic`` (Hess-Smith): the zero-length edge is skipped via
      the ``if d < 1e-14: continue`` guard.

    The trailing-edge pairs (``te_pairs``) are updated so that upper TE panels
    still reference the first-diagonal representation of the original upper TE
    quad, and the ``te_verts`` field is set from the original panel geometry so
    that ``build_panel_aic`` uses the generic TE path.

    Parameters
    ----------
    mesh : Mesh
        Any quad-panel mesh from ``make_wing_mesh``, ``make_vlm_mesh``, or a
        previously assembled mesh.

    Returns
    -------
    Mesh
        Mesh with the same number of panels but all stored as degenerate quads.
    """
    panels   = mesh.panels      # (Np, 4)
    vertices = mesh.vertices    # (Nv, 3)
    Np       = mesh.n_panels
    Nte      = len(mesh.te_pairs)

    # Build new panels: replace v3 index with v2 index
    new_panels = panels.copy()
    new_panels[:, 3] = new_panels[:, 2]    # (v0, v1, v2, v2)

    # ------------------------------------------------------------------
    # Compute te_verts from TE panel geometry (parametric convention:
    # upper TE edge is vertices v1–v2, i.e. aft-inner to aft-outer).
    # ------------------------------------------------------------------
    new_te_verts: np.ndarray | None = None
    if Nte > 0:
        if mesh.te_verts is not None:
            # Already available (imported mesh), copy as-is.
            new_te_verts = mesh.te_verts.copy()
        else:
            # Parametric mesh: derive from upper TE panel vertices.
            # Upper panels use ordering (fore-inner, aft-inner, aft-outer, fore-outer)
            # → TE edge is v1 (aft-inner) and v2 (aft-outer).
            new_te_verts = np.empty((Nte, 2, 3), dtype=float)
            te_u = mesh.te_pairs[:, 0]
            for k in range(Nte):
                v1_pt = vertices[panels[te_u[k], 1]]   # aft-inner
                v2_pt = vertices[panels[te_u[k], 2]]   # aft-outer
                # Sort inner/outer by |y| (inner = smaller |y|)
                if abs(v1_pt[1]) <= abs(v2_pt[1]):
                    new_te_verts[k, 0] = v1_pt   # inner (root-side)
                    new_te_verts[k, 1] = v2_pt   # outer (tip-side)
                else:
                    new_te_verts[k, 0] = v2_pt
                    new_te_verts[k, 1] = v1_pt

    # te_pairs: panel indices are unchanged (same Np, same index mapping)
    new_te_pairs = mesh.te_pairs.copy()

    # Copy wake_seed, surface_id, lifting_panels unchanged
    return Mesh(
        vertices=vertices,
        panels=new_panels,
        te_pairs=new_te_pairs,
        wake_seed=mesh.wake_seed.copy(),
        te_verts=new_te_verts,
        surface_id=mesh.surface_id.copy() if len(mesh.surface_id) > 0
                                          else np.empty(0, dtype=int),
        lifting_panels=mesh.lifting_panels.copy() if len(mesh.lifting_panels) > 0
                                                   else np.empty(0, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Mesh assembly
# ---------------------------------------------------------------------------

def find_tip_cap_panels(
    vertices: np.ndarray,
    triangles: np.ndarray,
    comp_ids: np.ndarray,
    lifting_comp_ids: list[int],
    ny_thresh: float = 0.5,
) -> np.ndarray:
    """Return a boolean mask of *tip-cap* panels on lifting surfaces.

    Tip-cap panels are identified as lifting-surface triangles whose outward
    normal has a large spanwise (y) component, i.e. ``|ny| > ny_thresh``.
    For OpenVSP wings without dihedral these panels have ny ≈ ±1; for wings
    with dihedral the threshold still discriminates them from aero panels.

    Parameters
    ----------
    vertices : (Nv, 3) float
    triangles : (Nt, 3) int  (0-based, already masked if needed)
    comp_ids : (Nt,) int
    lifting_comp_ids : list[int]
    ny_thresh : float
        Panels with ``|ny| > ny_thresh`` and belonging to a lifting component
        are labelled as tip-cap panels.  Default 0.5.

    Returns
    -------
    is_tip_cap : (Nt,) bool
    """
    lifting_set = set(lifting_comp_ids)

    v = vertices[triangles]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    raw_n = np.cross(e1, e2)
    mag   = np.linalg.norm(raw_n, axis=1, keepdims=True)
    normals = raw_n / np.where(mag > 0, mag, 1.0)   # (Nt, 3)

    is_lifting = np.array([c in lifting_set for c in comp_ids], dtype=bool)
    is_tip_cap = is_lifting & (np.abs(normals[:, 1]) > ny_thresh)
    return is_tip_cap


def tris_to_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    comp_ids: np.ndarray,
    lifting_comp_ids: list[int],
    body_comp_ids: list[int] | None = None,
    te_angle_thresh: float = 150.0,
    mask: np.ndarray | None = None,
    exclude_tip_cap: bool = True,
    ny_thresh: float = 0.5,
) -> Mesh:
    """Assemble a ``Mesh`` from Cart3D triangulated data.

    Triangles are stored as *degenerate quads* ``(i, j, k, k)`` in
    ``Mesh.panels``.  The ``Mesh.te_verts`` attribute is set from the
    detected TE edge endpoints so that ``build_panel_aic`` can construct
    the correct wake horseshoe without assuming the parametric vertex
    ordering.

    Parameters
    ----------
    vertices : (Nv, 3) float
    triangles : (Nt, 3) int  (0-based)
    comp_ids : (Nt,) int
    lifting_comp_ids : list[int]
        Component IDs of lifting surfaces (need Kutta condition + wake).
        Triangles from these components have ``lifting_panels = True``.
    body_comp_ids : list[int] or None
        Component IDs of non-lifting bodies (source panels only, no Kutta,
        no wake).  If None, all components not in ``lifting_comp_ids`` that
        are in ``mask`` are treated as body panels.
    te_angle_thresh : float
        Crease-angle threshold for TE detection (degrees).  Default 150°.
    mask : (Nt,) bool or None
        If provided, only the True-masked triangles are included in the Mesh.
        Useful to select a single wing half (e.g. ``comp_ids == 1``).
    exclude_tip_cap : bool
        If True (default), remove tip-cap panels from lifting surfaces before
        assembling the Mesh.  Tip-cap panels close the wing-tip surface and
        make the Mesh topologically closed, which produces a near-singular AIC
        (condition number ~ 10¹⁸ vs ~ 10⁶ without tip cap).  See ROADMAP §3
        (``include_tip_cap`` discussion) and ``find_tip_cap_panels``.
    ny_thresh : float
        Spanwise-normal threshold for tip-cap detection (default 0.5).

    Returns
    -------
    Mesh
        Valid Mesh with ``lifting_panels`` and ``te_verts`` set.
    """
    if mask is not None:
        triangles = triangles[mask]
        comp_ids  = comp_ids[mask]

    # Optionally remove tip-cap panels (prevents closed-body null-space)
    if exclude_tip_cap:
        tip_mask = find_tip_cap_panels(
            vertices, triangles, comp_ids, lifting_comp_ids, ny_thresh
        )
        if tip_mask.any():
            triangles = triangles[~tip_mask]
            comp_ids  = comp_ids[~tip_mask]

    # Detect TE edges *before* any vertex renumbering (original indices)
    te_pairs_orig, te_verts, wake_seed = detect_te_edges(
        vertices, triangles, comp_ids, lifting_comp_ids, te_angle_thresh
    )

    # Build mapping from original triangle index to local (post-mask) index
    # te_pairs_orig contains *local* indices already (triangles was already
    # masked above before passing to detect_te_edges).
    # (No remapping needed — detect_te_edges works on the masked arrays.)

    # Degenerate-quad panels: (i, j, k, k)
    panels = triangles[:, [0, 1, 2, 2]]       # (Nt, 4)

    # surface_id from comp_ids
    surface_id = comp_ids.copy()

    # lifting_panels flag
    lifting_set = set(lifting_comp_ids)
    lifting_panels = np.array([c in lifting_set for c in comp_ids], dtype=bool)

    return Mesh(
        vertices=vertices,
        panels=panels,
        te_pairs=te_pairs_orig,
        wake_seed=wake_seed,
        te_verts=te_verts,
        surface_id=surface_id,
        lifting_panels=lifting_panels,
    )
