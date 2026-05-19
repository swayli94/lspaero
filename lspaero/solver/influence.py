"""Aerodynamic influence coefficient (AIC) matrix for the VLM.

Each panel j contributes a horseshoe vortex of unit strength Γ=1:
  - Bound segment from A_j (1/4-chord inner) to B_j (1/4-chord outer)
  - Semi-infinite trailing leg from B_j in the wake direction d̂
  - Semi-infinite trailing leg from A_j in −d̂ (sign reversal for ∞→A leg)

Symmetry: the right half-wing is solved with an image system for the left half,
so AIC[i,j] includes contributions from both the right-wing and its image
horseshoe. This produces the correct induced velocities for a symmetric full wing
from only n_span×n_chord unknowns.

Reference: Katz & Plotkin §12.3, §13.3.
"""
from __future__ import annotations

import numpy as np

from ..geometry.mesh import Mesh
from .biot_savart import vel_segment, vel_semi_inf


def _ring_points(mesh: Mesh):
    """Extract per-panel vortex-ring points and collocation points.

    For each panel with corners [v0, v1, v2, v3] (v0=fwd-inner, v1=aft-inner,
    v2=aft-outer, v3=fwd-outer):

    - A  (1/4-chord inner)  = 0.75·v0 + 0.25·v1
    - B  (1/4-chord outer)  = 0.75·v3 + 0.25·v2
    - cp (3/4-chord midspan)= 0.125·v0 + 0.375·v1 + 0.375·v2 + 0.125·v3

    Returns
    -------
    A, B, cp : each (Np, 3)
    """
    v = mesh.vertices[mesh.panels]   # (Np, 4, 3)
    v0, v1, v2, v3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]

    A  = 0.75 * v0 + 0.25 * v1                                    # (Np, 3)
    B  = 0.75 * v3 + 0.25 * v2
    cp = 0.125 * v0 + 0.375 * v1 + 0.375 * v2 + 0.125 * v3      # 3/4-chord mid

    return A, B, cp


def build_aic(
    mesh: Mesh,
    wake_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the VLM aerodynamic influence coefficient matrix.

    AIC[i, j] = (induced normal-wash at collocation i) / Γ_j,
    where the horseshoe j includes both right-wing and mirror-image contributions
    (symmetric full-wing model).

    Parameters
    ----------
    mesh : Mesh
        VLM camber-surface mesh (produced by ``make_vlm_mesh``).
    wake_dir : (3,) array
        Unit vector giving the wake / trailing-filament direction
        (typically aligned with the freestream, e.g. [cos α, 0, sin α]).

    Returns
    -------
    AIC : (Np, Np) float
        Influence coefficient matrix.  Rows = collocation points.  Cols = panels.
    A : (Np, 3)
        1/4-chord inner bound-vortex endpoints (for K-J force computation).
    B : (Np, 3)
        1/4-chord outer bound-vortex endpoints.
    cp : (Np, 3)
        3/4-chord collocation points (where the no-penetration BC is applied).
    """
    A, B, cp = _ring_points(mesh)   # (Np, 3)
    Np = mesh.n_panels

    d = wake_dir / np.linalg.norm(wake_dir)   # ensure unit vector

    # Expand for vectorised (Np_i, Np_j, 3) computation
    P  = cp[:, None, :]    # (Np, 1, 3)
    _A = A[None, :, :]     # (1, Np, 3)
    _B = B[None, :, :]     # (1, Np, 3)

    # ---- Right-wing horseshoe ---- #
    vel = vel_segment(P, _A, _B)                         # (Np, Np, 3) bound
    vel = vel + vel_semi_inf(P, _B, d)                   # right trailing
    vel = vel - vel_semi_inf(P, _A, d)                   # left trailing (neg)

    # ---- Image horseshoe (mirror in y=0 plane) ---- #
    # Image bound runs B'→A' (same +y sense on the left half-wing, giving +lift)
    A_img = _A.copy(); A_img[..., 1] = -A_img[..., 1]
    B_img = _B.copy(); B_img[..., 1] = -B_img[..., 1]

    vel = vel + vel_segment(P, B_img, A_img)             # image bound B'→A'
    vel = vel + vel_semi_inf(P, A_img, d)                # image right trailing
    vel = vel - vel_semi_inf(P, B_img, d)                # image left trailing

    # ---- Dot with panel normals → AIC ---- #
    # n[i] dotted with vel[i, j, :] → AIC[i, j]
    AIC = np.einsum("ijk,ik->ij", vel, mesh.normals)    # (Np, Np)

    return AIC, A, B, cp


def induced_velocity_at(
    points: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Gamma: np.ndarray,
    wake_dir: np.ndarray,
    include_image: bool = True,
) -> np.ndarray:
    """Compute total induced velocity at arbitrary points from all horseshoes.

    Used after solving for Γ to evaluate the velocity field (e.g. for K-J
    force computation or wake visualisation).

    Parameters
    ----------
    points : (Nm, 3) evaluation points
    A, B : (Np, 3) bound-vortex endpoints (from ``build_aic``)
    Gamma : (Np,) panel circulations
    wake_dir : (3,)
    include_image : bool
        Include the symmetric image system.

    Returns
    -------
    vel : (Nm, 3)
    """
    Nm = len(points)
    Np = len(A)
    d = wake_dir / np.linalg.norm(wake_dir)

    P  = points[:, None, :]    # (Nm, 1, 3)
    _A = A[None, :, :]         # (1, Np, 3)
    _B = B[None, :, :]

    # Right-wing
    v = vel_segment(P, _A, _B) + vel_semi_inf(P, _B, d) - vel_semi_inf(P, _A, d)

    if include_image:
        A_img = _A.copy(); A_img[..., 1] = -A_img[..., 1]
        B_img = _B.copy(); B_img[..., 1] = -B_img[..., 1]
        v = (v
             + vel_segment(P, B_img, A_img)
             + vel_semi_inf(P, A_img, d)
             - vel_semi_inf(P, B_img, d))

    # Weighted sum: vel[m, :] = Σ_j Γ_j * v[m, j, :]
    vel = np.einsum("mji,j->mi", v.reshape(Nm, Np, 3), Gamma)
    return vel   # (Nm, 3)
