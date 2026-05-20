"""Aerodynamic influence coefficient (AIC) matrices for VLM and thick-panel methods.

VLM (build_aic):
  Each panel j contributes a horseshoe vortex of unit strength Γ=1:
    - Bound segment from A_j (1/4-chord inner) to B_j (1/4-chord outer)
    - Semi-infinite trailing leg from B_j in the wake direction d̂
    - Semi-infinite trailing leg from A_j in −d̂ (sign reversal for ∞→A leg)

Thick-panel vortex-ring method (build_panel_aic):
  Each panel j carries a closed vortex ring running around its four edges.
  Adjacent panels share edges — the shared edge vorticity cancels, leaving only
  the panel boundary vorticity. The Kutta condition is embedded by modifying the
  AIC columns for TE panels to account for semi-infinite wake horseshoes that carry
  the circulation jump Γ_upper − Γ_lower at each spanwise TE strip.

Symmetry: both methods model the full wing from a right half-wing mesh by adding
a mirror-image panel/horseshoe system (y → −y, reversed winding).

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001):
  VLM: §12.3, §13.3.
  Vortex-ring thick panel: §12.1, §12.5.
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


# ---------------------------------------------------------------------------
# Thick-surface vortex-ring AIC
# ---------------------------------------------------------------------------

def build_panel_aic(
    mesh: Mesh,
    wake_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the AIC matrix for the thick-surface vortex-ring panel method.

    Each panel j carries a unit-strength vortex ring around its four edges
    (v0→v1→v2→v3→v0, CCW from outside so the ring normal matches the panel
    outward normal).  A mirror-image system (y → −y, reversed winding) models
    the left half-wing for symmetric flight.

    The Kutta condition is enforced by adding the influence of semi-infinite
    wake horseshoes that shed from each TE strip with strength Γ_u − Γ_l.
    This modifies the AIC columns for the upper and lower TE panels so that
    the linear system already embeds the Kutta constraint.

    Parameters
    ----------
    mesh : Mesh
        Thick-surface mesh from ``make_wing_mesh``.
    wake_dir : (3,) array
        Unit wake direction (usually aligned with the freestream).

    Returns
    -------
    AIC : (Np, Np) float
        Effective AIC with Kutta condition embedded.
    vel_body : (Np, Np, 3) float
        Velocity tensor: vel_body[i, j, :] is the velocity at collocation i
        from a unit-strength ring j (right wing + image).  Used post-solve to
        compute surface velocity and Cp.
    wake_vel : (Np, Nte, 3) float
        Wake horseshoe velocity tensor (right + image): wake_vel[i, k, :] is
        the velocity at collocation i from a unit-strength wake strip k.
    A_w : (Nte, 3) float
        Inner (root-side) TE midpoint per spanwise strip — wake bound start.
    B_w : (Nte, 3) float
        Outer (tip-side) TE midpoint per spanwise strip — wake bound end.

    Notes
    -----
    Reference: Katz & Plotkin §12.1 (vortex-ring method) and §12.5 (Cp from
    surface velocity).
    """
    Np = mesh.n_panels
    pv = mesh.vertices[mesh.panels]       # (Np, 4, 3)
    cp = mesh.centroids                   # (Np, 3)
    n  = mesh.normals                     # (Np, 3)
    d  = wake_dir / np.linalg.norm(wake_dir)

    # Broadcast shapes: P eval (Np, 1, 3), ring verts (1, Np, 3)
    P  = cp[:, None, :]
    V0 = pv[None, :, 0, :]
    V1 = pv[None, :, 1, :]
    V2 = pv[None, :, 2, :]
    V3 = pv[None, :, 3, :]

    # Right-wing ring velocity: v0→v1→v2→v3→v0
    vel_body = (vel_segment(P, V0, V1) + vel_segment(P, V1, V2) +
                vel_segment(P, V2, V3) + vel_segment(P, V3, V0))  # (Np, Np, 3)

    # Image ring: mirror y→-y, reversed winding v0'→v3'→v2'→v1'→v0'
    pv_img = pv.copy()
    pv_img[..., 1] = -pv_img[..., 1]
    V0i = pv_img[None, :, 0, :]
    V1i = pv_img[None, :, 1, :]
    V2i = pv_img[None, :, 2, :]
    V3i = pv_img[None, :, 3, :]
    vel_body += (vel_segment(P, V0i, V3i) + vel_segment(P, V3i, V2i) +
                 vel_segment(P, V2i, V1i) + vel_segment(P, V1i, V0i))

    # AIC normal component from body rings
    AIC = np.einsum("ijk,ik->ij", vel_body, n)   # (Np, Np)

    # ---- Wake geometry from TE pairs ----
    # Upper TE panel winding: v0=fore-inner, v1=aft-inner, v2=aft-outer, v3=fore-outer
    # Lower TE panel winding: v0=fore-inner, v1=fore-outer, v2=aft-outer, v3=aft-inner
    # TE edge: upper v1(aft-inner)→v2(aft-outer); lower v2(aft-outer)→v3(aft-inner)
    te_u = mesh.te_pairs[:, 0]   # (Nte,) upper TE panel indices
    te_l = mesh.te_pairs[:, 1]   # (Nte,) lower TE panel indices
    pu = mesh.vertices[mesh.panels[te_u]]  # (Nte, 4, 3)
    pl = mesh.vertices[mesh.panels[te_l]]

    A_w = 0.5 * (pu[:, 1, :] + pl[:, 3, :])   # inner TE midpoint  (Nte, 3)
    B_w = 0.5 * (pu[:, 2, :] + pl[:, 2, :])   # outer TE midpoint  (Nte, 3)

    # Wake horseshoe: bound A_w→B_w (inner→outer) + two semi-infinite legs
    Pp   = cp[:, None, :]     # (Np, 1, 3)
    _Aw  = A_w[None, :, :]   # (1, Nte, 3)
    _Bw  = B_w[None, :, :]

    wake_vel = (vel_segment(Pp, _Aw, _Bw) +
                vel_semi_inf(Pp, _Bw, d) -
                vel_semi_inf(Pp, _Aw, d))   # (Np, Nte, 3)

    # Image wake: mirror y→-y, bound runs B_w'→A_w' (reversed for left wing)
    _Aw_img = _Aw.copy(); _Aw_img[..., 1] = -_Aw_img[..., 1]
    _Bw_img = _Bw.copy(); _Bw_img[..., 1] = -_Bw_img[..., 1]
    wake_vel += (vel_segment(Pp, _Bw_img, _Aw_img) +
                 vel_semi_inf(Pp, _Aw_img, d) -
                 vel_semi_inf(Pp, _Bw_img, d))

    # ---- Embed Kutta condition into AIC ----
    # wake_AIC[i, k] = normal component of unit-wake-strip-k velocity at panel i
    wake_AIC = np.einsum("ijk,ik->ij", wake_vel, n)   # (Np, Nte)
    # Upper TE panel column: add wake influence (Γ_wake = Γ_u - Γ_l → coeff +1)
    # Lower TE panel column: subtract wake influence (coeff -1)
    np.add.at(AIC, (slice(None), te_u), wake_AIC)
    np.subtract.at(AIC, (slice(None), te_l), wake_AIC)

    return AIC, vel_body, wake_vel, A_w, B_w


# ---------------------------------------------------------------------------
# Flexible-wake AIC (for free-wake / wake-relaxation VLM)
# ---------------------------------------------------------------------------

def build_aic_with_wake(
    mesh: Mesh,
    nodes_A: np.ndarray,
    nodes_B: np.ndarray,
    d_far: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VLM AIC built with a finite wake node system instead of semi-infinite legs.

    Each horseshoe vortex's trailing legs are replaced by a series of finite
    vortex segments (one per wake row) plus a single semi-infinite extension
    from the last row.  When ``nodes_A / nodes_B`` are collinear with
    ``d_far`` (the initial planar configuration), the result is identical to
    ``build_aic(mesh, d_far)`` by the Biot–Savart telescoping identity.

    Parameters
    ----------
    mesh : Mesh
        Camber-surface VLM mesh.
    nodes_A, nodes_B : (n_rows+1, Np, 3)
        Wake node arrays from ``build_wake_nodes``.  Row 0 must equal A / B
        from ``_ring_points(mesh)``.
    d_far : (3,)
        Far-field direction for the semi-infinite extension from the last row.
        Typically the freestream unit vector.

    Returns
    -------
    AIC : (Np, Np)
    A, B, cp : (Np, 3)  — bound-vortex geometry for K-J force computation.

    Notes
    -----
    Katz & Plotkin §14.2: free-wake horseshoe vortex system.
    """
    A, B, cp = _ring_points(mesh)
    n_rows = nodes_A.shape[0] - 1
    d = d_far / np.linalg.norm(d_far)

    P  = cp[:, None, :]    # (Np, 1, 3)  collocation points

    # ---- Bound segment ---- #
    vel = vel_segment(P, A[None, :, :], B[None, :, :])   # (Np, Np, 3)

    # ---- Right trailing legs: B[r] → B[r+1] then semi-infinite ---- #
    for r in range(n_rows):
        nB_r  = nodes_B[r,     :, :][None, :, :]   # (1, Np, 3)
        nB_r1 = nodes_B[r + 1, :, :][None, :, :]
        vel = vel + vel_segment(P, nB_r, nB_r1)
    vel = vel + vel_semi_inf(P, nodes_B[-1, :, :][None, :, :], d)

    # ---- Left trailing legs: A[r] → A[r+1] then semi-infinite (minus) ---- #
    for r in range(n_rows):
        nA_r  = nodes_A[r,     :, :][None, :, :]
        nA_r1 = nodes_A[r + 1, :, :][None, :, :]
        vel = vel - vel_segment(P, nA_r, nA_r1)
    vel = vel - vel_semi_inf(P, nodes_A[-1, :, :][None, :, :], d)

    # ---- Image system (mirror y → -y, reversed winding for left half-wing) ---- #
    # Row 0 of nodes equals A / B, so the image bound is automatically correct.
    nodes_A_img = nodes_A.copy();  nodes_A_img[:, :, 1] *= -1
    nodes_B_img = nodes_B.copy();  nodes_B_img[:, :, 1] *= -1

    nA0i = nodes_A_img[0, :, :][None, :, :]   # (1, Np, 3) = A'
    nB0i = nodes_B_img[0, :, :][None, :, :]   # = B'

    # Image bound runs B'→A' (same +y sense on the left half-wing)
    vel = vel + vel_segment(P, nB0i, nA0i)

    # Image right trailing: from A' (plus sign, toward root)
    for r in range(n_rows):
        nAi_r  = nodes_A_img[r,     :, :][None, :, :]
        nAi_r1 = nodes_A_img[r + 1, :, :][None, :, :]
        vel = vel + vel_segment(P, nAi_r, nAi_r1)
    vel = vel + vel_semi_inf(P, nodes_A_img[-1, :, :][None, :, :], d)

    # Image left trailing: from B' (minus sign)
    for r in range(n_rows):
        nBi_r  = nodes_B_img[r,     :, :][None, :, :]
        nBi_r1 = nodes_B_img[r + 1, :, :][None, :, :]
        vel = vel - vel_segment(P, nBi_r, nBi_r1)
    vel = vel - vel_semi_inf(P, nodes_B_img[-1, :, :][None, :, :], d)

    AIC = np.einsum("ijk,ik->ij", vel, mesh.normals)
    return AIC, A, B, cp
