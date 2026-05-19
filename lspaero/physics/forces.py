"""Kutta-Joukowski force and moment integration for the VLM.

Near-field force formula (Katz & Plotkin §12.5, Eq. 12.28):

    F_j = ρ · Γ_j · (V_total_j × ΔL_j)

where V_total_j = V_∞ + V_induced_j at the bound-vortex midpoint of panel j,
and ΔL_j = B_j − A_j is the bound-vortex vector.

The lift L is the component of ΣF_j perpendicular to V_∞ in the symmetry plane.
The induced drag D_i is the component parallel to V_∞.
The pitching moment M is taken about x_ref.

Non-dimensionalisation uses the *half-wing* reference area (S_ref = Σ panel
areas on the right half) so that CL, CDi, Cm computed from the half-wing match
the full-wing values directly.  (Factor of 2 cancels: 2·L_half / (q·2·S_half).)
"""
from __future__ import annotations

import numpy as np


def compute_forces(
    A: np.ndarray,
    B: np.ndarray,
    Gamma: np.ndarray,
    V_inf: np.ndarray,
    V_induced: np.ndarray,
    rho: float,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    x_ref: float = 0.25,
) -> dict:
    """Compute CL, CDi, Cm from K-J integration.

    Parameters
    ----------
    A, B : (Np, 3)
        Bound-vortex endpoints (1/4-chord inner / outer).
    Gamma : (Np,)
        Panel circulations (from VLM solve).
    V_inf : (3,)
        Freestream velocity vector.
    V_induced : (Np, 3)
        Induced velocity at each bound-segment midpoint.
    rho : float
        Air density (kg/m³).
    S_ref : float
        Reference area (half-wing planform area, m²).
    b_ref : float
        Reference span (full span = 2 × half_span, m).
    c_ref : float
        Reference chord (MAC, m).
    x_ref : float
        x-coordinate of moment reference point (m, default = 0.25·c_ref
        from LE, i.e. the standard aerodynamic centre location).

    Returns
    -------
    dict with keys: CL, CDi, Cm, L, Di, M, forces (Np, 3), dCL (per panel)
    """
    V_mag = float(np.linalg.norm(V_inf))
    q = 0.5 * rho * V_mag ** 2

    V_hat = V_inf / V_mag
    # Lift direction: perpendicular to V_inf in the symmetry (x-z) plane
    # For V_inf = [cos α, 0, sin α]: ê_L = [-sin α, 0, cos α]
    e_lift = np.array([-V_hat[2], 0.0, V_hat[0]])
    e_lift /= np.linalg.norm(e_lift) if np.linalg.norm(e_lift) > 1e-12 else 1.0

    dL = B - A                             # (Np, 3) bound-vortex vectors
    V_total = V_inf[None, :] + V_induced   # (Np, 3)

    # F_j = ρ · Γ_j · (V_total_j × ΔL_j)
    forces = rho * Gamma[:, None] * np.cross(V_total, dL)   # (Np, 3)

    F_total = forces.sum(axis=0)   # (3,)

    L  = float(F_total @ e_lift)
    Di = float(F_total @ V_hat)

    # Pitching moment about (x_ref, 0, 0)
    r_mid = 0.5 * (A + B)                        # bound-vortex midpoints (Np, 3)
    r_ref = r_mid - np.array([x_ref, 0.0, 0.0])
    moments = np.cross(r_ref, forces)             # (Np, 3)
    M = float(moments.sum(axis=0)[1])             # y-component (pitch up positive)

    CL  = L  / (q * S_ref)
    CDi = Di / (q * S_ref)
    Cm  = M  / (q * S_ref * c_ref)

    # Spanwise loading dCL for each panel (useful for plotting)
    l_j = np.einsum("ij,j->i", forces, e_lift)   # (Np,) lift per panel
    dCL = l_j / (q * S_ref)

    return {
        "CL": CL,
        "CDi": CDi,
        "Cm": Cm,
        "L": L,
        "Di": Di,
        "M": M,
        "forces": forces,
        "dCL": dCL,
    }
