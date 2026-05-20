"""Thick-surface vortex-ring panel solver.

Solves the no-penetration boundary value problem on a closed wing surface
(upper, lower, tip cap) using vortex rings, then computes surface Cp via
Bernoulli and integrates forces.

Typical usage::

    from lspaero.geometry.wing import make_wing_mesh
    from lspaero.solver.solve_panel import solve_panel

    mesh   = make_wing_mesh(half_span=5, root_chord=2, n_span=16, n_chord=6)
    result = solve_panel(mesh, alpha_deg=5.0)
    print(result["CL"], result["CDi"])
    print(result["Cp"])          # (Np,) pressure coefficient on each panel

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §12.1–12.5.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..geometry.mesh import Mesh
from ..physics.forces import forces_from_cp
from .influence import build_panel_aic


def solve_panel(
    mesh: Mesh,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    V_mag: float = 1.0,
    rho: float = 1.225,
    S_ref: float | None = None,
    b_ref: float | None = None,
    c_ref: float | None = None,
    x_ref: float | None = None,
) -> dict:
    """Solve the thick-surface vortex-ring panel method.

    The right half-wing mesh is completed by a symmetric image system so that
    all results correspond to the full symmetric wing.

    Algorithm (Katz & Plotkin §12.1):
    1. Build the vortex-ring AIC matrix with the Kutta condition embedded —
       the upper/lower TE panel columns are modified to include the influence
       of the semi-infinite wake that sheds with strength Γ_u − Γ_l.
    2. Solve AIC · Γ = −(V_∞ · n̂) for panel circulations.
    3. Evaluate total surface velocity:
           V_i = V_∞ + Σ_j Γ_j · vel_ring(c_i, j)
                      + Σ_k (Γ_u(k)−Γ_l(k)) · vel_wake(c_i, k)
    4. Bernoulli: Cp_i = 1 − |V_i|²/V_∞²
    5. Integrate: F = −q ∫ Cp n̂ dS → CL, CDi, Cm.

    Parameters
    ----------
    mesh : Mesh
        Thick-surface mesh from ``make_wing_mesh`` (upper + lower + tip cap).
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Sideslip angle in degrees.
    V_mag : float
        Freestream speed (any consistent unit system).
    rho : float
        Air density (kg/m³ or matching unit system).
    S_ref : float, optional
        Reference area (half-wing panel area sum if not provided).
    b_ref : float, optional
        Reference span (2 × max(y) if not provided).
    c_ref : float, optional
        Reference chord (S_ref / (0.5 × b_ref) if not provided).
    x_ref : float, optional
        Moment reference x-coordinate (0.25 × c_ref if not provided).

    Returns
    -------
    dict
        Keys: CL, CDi, Cm, L, Di, M,
              Gamma (Np,), Cp (Np,), V_surface (Np, 3),
              forces (Np, 3), V_inf (3,), wake_dir (3,),
              A_w (Nte, 3), B_w (Nte, 3),
              S_ref, b_ref, c_ref.
    """
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)

    V_inf = V_mag * np.array([
        np.cos(alpha) * np.cos(beta),
        -np.sin(beta),
        np.sin(alpha),
    ])
    wake_dir = V_inf / np.linalg.norm(V_inf)

    # Reference quantities
    if S_ref is None:
        S_ref = float(mesh.areas.sum())
    if b_ref is None:
        b_ref = 2.0 * float(mesh.vertices[:, 1].max())
    if c_ref is None:
        c_ref = S_ref / (0.5 * b_ref)
    if x_ref is None:
        x_ref = 0.25 * c_ref

    # ---- Build AIC and velocity tensors ----
    AIC, vel_body, wake_vel, A_w, B_w = build_panel_aic(mesh, wake_dir)

    # ---- No-penetration RHS ----
    rhs = -(V_inf @ mesh.normals.T)   # (Np,)

    # ---- Solve ----
    lu, piv = lu_factor(AIC)
    Gamma = lu_solve((lu, piv), rhs)   # (Np,)

    # ---- Surface velocity and Cp ----
    te_u = mesh.te_pairs[:, 0]          # (Nte,)
    te_l = mesh.te_pairs[:, 1]
    Gamma_wake = Gamma[te_u] - Gamma[te_l]   # (Nte,)

    # V_ind_body[i] = Σ_j Γ_j * vel_body[i, j, :]
    V_ind_body = np.einsum("ijk,j->ik", vel_body, Gamma)           # (Np, 3)
    # V_ind_wake[i] = Σ_k Γ_wake[k] * wake_vel[i, k, :]
    V_ind_wake = np.einsum("ijk,j->ik", wake_vel, Gamma_wake)      # (Np, 3)

    V_surface = V_inf[None, :] + V_ind_body + V_ind_wake           # (Np, 3)

    Cp = 1.0 - np.einsum("ij,ij->i", V_surface, V_surface) / (V_mag ** 2)

    # ---- Force integration: Cp path ----
    result = forces_from_cp(mesh, Cp, V_inf, rho, S_ref, b_ref, c_ref, x_ref)

    # ---- Force integration: wake K-J path (topology-independent) ----
    #
    # Kutta-Joukowski on the right-half wake horseshoes:
    #   F_k = ρ Γ_wake[k] (V_∞ × ΔB_k)
    # where Γ_wake[k] = Γ[te_u[k]] − Γ[te_l[k]] and ΔB_k = B_w[k] − A_w[k].
    #
    # This formulation is mesh-topology-independent: it works for structured quads,
    # degenerate-quad (triangulated), and imported Cart3D meshes alike.
    # It gives the correct full-wing CL when normalised by the half-wing S_ref
    # (the factor of 2 for the image wing cancels with the factor of 2 in
    # q·2·S_ref, identical to the convention in compute_forces).
    dB = B_w - A_w                              # (Nte, 3) wake bound-vortex vectors
    F_wake = rho * np.einsum(
        "k,ki->i",
        Gamma_wake,
        np.cross(np.broadcast_to(V_inf, dB.shape), dB),
    )                                           # (3,) right-half-wing wake force

    q          = 0.5 * rho * V_mag ** 2
    lift_hat   = np.array([-np.sin(alpha), 0.0,  np.cos(alpha)])
    drag_hat   = np.array([ np.cos(alpha), 0.0,  np.sin(alpha)])

    result.update({
        "Gamma":       Gamma,
        "Cp":          Cp,
        "V_surface":   V_surface,
        "V_inf":       V_inf,
        "wake_dir":    wake_dir,
        "A_w":         A_w,
        "B_w":         B_w,
        "S_ref":       S_ref,
        "b_ref":       b_ref,
        "c_ref":       c_ref,
        "CL_wake":     float(F_wake @ lift_hat) / (q * S_ref),
        "CDi_wake":    float(F_wake @ drag_hat) / (q * S_ref),
    })
    return result
