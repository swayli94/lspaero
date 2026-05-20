"""Mesh data structure for the LSPAero panel solver.

Both the parametric wing generator (geometry/wing.py) and the SDF-based
sampler (geometry/sdf/) produce this exact structure — the solver knows nothing
about where the mesh came from.

Coordinate convention: x chordwise (LE→TE), y spanwise (root→tip), z up.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §10.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    """Panel mesh consumed by the LSPAero solver.

    Geometry and topology are fixed at construction; derived quantities
    (centroids, normals, areas) are computed immediately and cached.

    Attributes
    ----------
    vertices : (Nv, 3) float
        Panel corner coordinates in the body frame.
    panels : (Np, 4) int
        Vertex-index quads, CCW when viewed from outside the surface so that
        the right-hand-rule normal points outward.
    te_pairs : (Nte, 2) int
        ``[upper_panel_idx, lower_panel_idx]`` for each spanwise trailing-edge
        strip.  Used to enforce the Kutta condition.
    wake_seed : (Nte, 3) float
        Midpoint of the upper/lower TE gap per strip — starting point for
        wake filaments.
    surface_id : (Np,) int, optional
        Integer label per panel: 0 = upper, 1 = lower, 2 = tip cap.
    centroids : (Np, 3) float  [computed]
        Mean of the four corner vertices.
    normals : (Np, 3) float  [computed]
        Unit outward normal via the diagonal cross-product formula:
        n ∝ (v₂ − v₀) × (v₃ − v₁).
    areas : (Np,) float  [computed]
        Panel area = ½ |(v₂ − v₀) × (v₃ − v₁)|  (exact for parallelograms).
    """

    # ------------------------------------------------------------------ #
    # Primary inputs                                                       #
    # ------------------------------------------------------------------ #
    vertices: np.ndarray
    panels: np.ndarray
    te_pairs: np.ndarray
    wake_seed: np.ndarray
    surface_id: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    # Stage 7 additions — optional, default-None / default-empty
    te_verts: np.ndarray | None = field(default=None)
    """(Nte, 2, 3) float — inner/outer TE edge endpoints per TE pair.

    When set (imported tri meshes), ``build_panel_aic`` uses these directly
    instead of inferring wake endpoints from the parametric panel vertex
    ordering.  None → fall back to the parametric formula.
    """
    lifting_panels: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    """(Np,) bool — True for panels that carry circulation (Kutta + wake).

    Defaults to all-True when empty.  Used in Stage 8 to exclude non-lifting
    body panels from wake generation.
    """

    # ------------------------------------------------------------------ #
    # Derived (set by __post_init__, not passed to the constructor)        #
    # ------------------------------------------------------------------ #
    centroids: np.ndarray = field(init=False, repr=False)
    normals: np.ndarray = field(init=False, repr=False)
    areas: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._check_index_bounds()
        self._compute_derived()
        self._check_invariants()

    # ------------------------------------------------------------------ #
    # Derived-quantity computation (fully vectorised)                     #
    # ------------------------------------------------------------------ #

    def _compute_derived(self) -> None:
        v = self.vertices[self.panels]           # (Np, 4, 3)

        # Centroid: degenerate quads (triangles with v[2]==v[3]) must use the
        # 3-vertex average; regular quads use the 4-vertex average.
        # Vectorised: identify degenerate panels, then blend.
        is_tri = (self.panels[:, 2] == self.panels[:, 3])   # (Np,) bool
        cent_quad = v.mean(axis=1)                           # (Np, 3)
        cent_tri  = v[:, :3].mean(axis=1)                   # (Np, 3)
        self.centroids = np.where(is_tri[:, None], cent_tri, cent_quad)

        # Normal and area via diagonal cross-product — correct for both quads
        # and degenerate quads:
        #   d1×d2 = (v2-v0)×(v3-v1)
        # For (i,j,k,k): d1=(k-i), d2=(k-j) → cross = (b-a)×(c-a)  [triangle]
        d1 = v[:, 2] - v[:, 0]                  # diagonal 0 → 2  (Np, 3)
        d2 = v[:, 3] - v[:, 1]                  # diagonal 1 → 3  (Np, 3)
        cross = np.cross(d1, d2)                 # (Np, 3)
        mag = np.linalg.norm(cross, axis=1)      # (Np,)
        self.normals = cross / mag[:, None]      # (Np, 3) unit normals
        self.areas = 0.5 * mag                   # (Np,)

    # ------------------------------------------------------------------ #
    # Invariant checks (structural; geometry checks are optional methods) #
    # ------------------------------------------------------------------ #

    def _check_index_bounds(self) -> None:
        Nv = len(self.vertices)
        if self.panels.min() < 0 or self.panels.max() >= Nv:
            raise ValueError("panels contains out-of-range vertex indices")

    def _check_invariants(self) -> None:
        Np = len(self.panels)
        if self.te_pairs.size > 0:
            if self.te_pairs.min() < 0 or self.te_pairs.max() >= Np:
                raise ValueError("te_pairs contains out-of-range panel indices")
        if not np.all(self.areas > 0):
            raise ValueError("degenerate panels detected (zero or negative area)")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_panels(self) -> int:
        return len(self.panels)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    # ------------------------------------------------------------------ #
    # Optional geometric checks (call explicitly; not enforced at init)   #
    # ------------------------------------------------------------------ #

    def is_watertight(self, tol: float = 1e-4) -> bool:
        """Divergence-theorem check: Σ area·normal ≈ 0 for closed surfaces."""
        residual = (self.normals * self.areas[:, None]).sum(axis=0)
        return float(np.linalg.norm(residual)) < tol

    def max_non_planarity(self) -> float:
        """Maximum deviation of the 4th vertex from the plane of the first 3.

        Zero for perfectly planar panels; grows for warped quads.  Useful for
        diagnosing mesh quality on swept/dihedral wings.
        """
        v = self.vertices[self.panels]   # (Np, 4, 3)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        e3 = v[:, 3] - v[:, 0]
        n = np.cross(e1, e2)
        n_mag = np.linalg.norm(n, axis=1, keepdims=True)
        n_hat = n / np.where(n_mag > 0, n_mag, 1.0)
        return float(np.max(np.abs(np.einsum("ij,ij->i", e3, n_hat))))


# ---------------------------------------------------------------------------
# Multi-body merge
# ---------------------------------------------------------------------------

def combine_meshes(meshes: list["Mesh"]) -> "Mesh":
    """Merge a list of Mesh objects into a single combined Mesh.

    Vertex arrays are stacked with appropriate index offsets; ``te_pairs``,
    ``wake_seed``, ``surface_id``, ``lifting_panels``, and ``te_verts`` are
    concatenated.  The combined mesh is the sole object passed to the solver —
    it carries no information about individual component origins.

    Typical use::

        wing = make_wing_mesh(...)
        body = make_body_mesh(...)
        combined = combine_meshes([wing, body])
        result = solve_morino(combined, cam_mesh=make_vlm_mesh(...))

    Parameters
    ----------
    meshes : list[Mesh]
        Component meshes to merge.  At least one must be provided.

    Returns
    -------
    Mesh
        Combined mesh.  Invariants:

        * Total vertex count = Σ n_vertices.
        * Total panel count  = Σ n_panels.
        * Panel indices in the combined ``panels`` array are offset so that
          they index into the combined ``vertices`` array.
        * ``te_pairs`` indices are offset so that they index into the combined
          ``panels`` array.
        * ``lifting_panels`` is ``True`` for panels from meshes whose
          ``lifting_panels`` was all-True (or empty, i.e. default); ``False``
          for explicitly non-lifting panels (e.g. fuselage from
          ``make_body_mesh``).
    """
    if not meshes:
        raise ValueError("combine_meshes requires at least one Mesh.")

    all_vertices: list[np.ndarray]      = []
    all_panels:   list[np.ndarray]      = []
    all_te_pairs: list[np.ndarray]      = []
    all_wake_seed: list[np.ndarray]     = []
    all_surface_id: list[np.ndarray]    = []
    all_lifting:   list[np.ndarray]     = []
    all_te_verts:  list[np.ndarray]     = []
    has_te_verts = False

    v_off = 0   # running vertex index offset
    p_off = 0   # running panel index offset

    for m in meshes:
        Np = m.n_panels
        Nv = m.n_vertices

        all_vertices.append(m.vertices)
        all_panels.append(m.panels + v_off)

        if m.te_pairs.size > 0:
            all_te_pairs.append(m.te_pairs + p_off)
        if m.wake_seed.shape[0] > 0:
            all_wake_seed.append(m.wake_seed)

        # surface_id: fall back to all-zero if missing
        sid = m.surface_id if m.surface_id.size > 0 else np.zeros(Np, dtype=int)
        all_surface_id.append(sid)

        # lifting_panels: empty → all lifting (backward compat)
        lp = m.lifting_panels
        if lp.size == 0:
            lp = np.ones(Np, dtype=bool)
        all_lifting.append(lp)

        # te_verts (optional)
        if m.te_verts is not None:
            all_te_verts.append(m.te_verts)
            has_te_verts = True
        else:
            all_te_verts.append(None)

        v_off += Nv
        p_off += Np

    combined_vertices  = np.vstack(all_vertices)
    combined_panels    = np.vstack(all_panels)

    if all_te_pairs:
        combined_te_pairs  = np.vstack(all_te_pairs)
    else:
        combined_te_pairs  = np.empty((0, 2), dtype=int)

    if all_wake_seed:
        combined_wake_seed = np.vstack(all_wake_seed)
    else:
        combined_wake_seed = np.empty((0, 3), dtype=float)

    combined_surface_id = np.concatenate(all_surface_id)
    combined_lifting    = np.concatenate(all_lifting)

    combined_te_verts: np.ndarray | None = None
    if has_te_verts:
        # Only include te_verts from meshes that have them; others have none.
        # (Typically only imported-tri wing meshes carry te_verts.)
        parts = [tv for tv in all_te_verts if tv is not None]
        if parts:
            combined_te_verts = np.vstack(parts)

    return Mesh(
        vertices=combined_vertices,
        panels=combined_panels,
        te_pairs=combined_te_pairs,
        wake_seed=combined_wake_seed,
        surface_id=combined_surface_id,
        te_verts=combined_te_verts,
        lifting_panels=combined_lifting,
    )
