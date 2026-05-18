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
        v = self.vertices[self.panels]          # (Np, 4, 3)
        self.centroids = v.mean(axis=1)          # (Np, 3)
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
