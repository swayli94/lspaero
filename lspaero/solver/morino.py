"""Thick-surface aerodynamic solver: VLM + Hess-Smith source panels.

The public entry point is ``solve_morino``, which combines two complementary
panel-method solves to produce accurate global forces and physically correct
surface Cp on a thick (upper + lower surface) wing mesh.

Algorithm
---------
1. **Hess-Smith source solve on the thick surface**:
   Solves AIC_σ · σ = −(V_∞ · n̂) for source strengths that enforce zero
   normal velocity.  Tangential surface velocity → Cp_thickness on each panel.
   The self-panel normal singularity is removed by projecting out the normal
   component: Cp = 1 − |V_t|²/V_∞² where V_t = V − (V·n̂)n̂.

2. **VLM on the mean camber surface** (vertex-averaged from the thick mesh):
   Solves for panel circulations Γ and computes Kutta-Joukowski lift, induced
   drag, and pitching moment.  This is the *primary* force output.
   Also provides ΔCp_j = 2Γ_j / (V_∞ Δx_j) per camber panel.

3. **Superimpose** thickness and lifting effects:
   Cp_upper[j] = Cp_thickness_upper[j] − ΔCp_j / 2   (suction)
   Cp_lower[j] = Cp_thickness_lower[j] + ΔCp_j / 2   (pressure)

This module also retains helpers for the potential-form Morino BIE
(doublet/source AIC builders, wake Kutta embedding) that were used during
development and may be useful for research.

References
----------
Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §10.2, §12.1–12.5.
"""
from __future__ import annotations

import numpy as np

from ..geometry.mesh import Mesh
from ..physics.forces import forces_from_cp


# ---------------------------------------------------------------------------
# Solid-angle helpers
# ---------------------------------------------------------------------------

def _solid_angle_triangle(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
) -> np.ndarray:
    """Solid angle of triangle (r1,r2,r3) from origin (Van Oosterom 1983).

    Parameters
    ----------
    r1, r2, r3 : (..., 3)  vertex position vectors from the eval point.

    Returns
    -------
    omega : (...,)  solid angle in steradians (signed).
    """
    m1 = np.linalg.norm(r1, axis=-1)
    m2 = np.linalg.norm(r2, axis=-1)
    m3 = np.linalg.norm(r3, axis=-1)

    num = np.einsum("...i,...i->...", r1, np.cross(r2, r3))
    den = (m1 * m2 * m3
           + np.einsum("...i,...i->...", r1, r2) * m3
           + np.einsum("...i,...i->...", r2, r3) * m1
           + np.einsum("...i,...i->...", r3, r1) * m2)

    return 2.0 * np.arctan2(num, np.abs(den) + 1e-300) * np.sign(den + 1e-300 * (den == 0))


def _solid_angle_quad(cp: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """Solid angle of a quadrilateral panel at eval points.

    Parameters
    ----------
    cp  : (Ni, 3) eval centroids
    pv  : (4, 3)  panel vertices

    Returns
    -------
    omega : (Ni,)
    """
    r0 = cp - pv[0]   # (Ni, 3)
    r1 = cp - pv[1]
    r2 = cp - pv[2]
    r3 = cp - pv[3]
    return _solid_angle_triangle(r0, r1, r2) + _solid_angle_triangle(r0, r2, r3)


# ---------------------------------------------------------------------------
# Gauss quadrature on a bilinear quad panel
# ---------------------------------------------------------------------------

def _gauss_sample_quad(pv: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss sample points and weights on a bilinear quad panel.

    Parameters
    ----------
    pv : (4, 3) panel vertices (v0, v1, v2, v3 in CCW order).
    n  : number of Gauss points per direction (2, 3, or 4).

    Returns
    -------
    Q   : (n², 3) sample positions in 3-D.
    wJ  : (n²,)  weights * Jacobian magnitude.
    """
    if n == 2:
        xi1d = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        w1d  = np.array([1.0, 1.0])
    elif n == 3:
        xi1d = np.array([-np.sqrt(3.0 / 5), 0.0, np.sqrt(3.0 / 5)])
        w1d  = np.array([5.0 / 9, 8.0 / 9, 5.0 / 9])
    else:  # n == 4
        c = np.sqrt(3.0 / 7)
        xi1d = np.array([-np.sqrt(3.0/7 + 2/7*np.sqrt(6.0/5)),
                          -np.sqrt(3.0/7 - 2/7*np.sqrt(6.0/5)),
                           np.sqrt(3.0/7 - 2/7*np.sqrt(6.0/5)),
                           np.sqrt(3.0/7 + 2/7*np.sqrt(6.0/5))])
        w1d = np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36,
                         (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36])

    xi2d, eta2d = np.meshgrid(xi1d, xi1d, indexing='ij')
    w2d = (w1d[:, None] * w1d[None, :]).ravel()
    xi2d = xi2d.ravel(); eta2d = eta2d.ravel()

    v0, v1, v2, v3 = pv
    N0 = 0.25 * (1 - xi2d) * (1 - eta2d)
    N1 = 0.25 * (1 + xi2d) * (1 - eta2d)
    N2 = 0.25 * (1 + xi2d) * (1 + eta2d)
    N3 = 0.25 * (1 - xi2d) * (1 + eta2d)
    Q = N0[:, None]*v0 + N1[:, None]*v1 + N2[:, None]*v2 + N3[:, None]*v3

    dPxi  = 0.25 * (-(1-eta2d)[:,None]*v0 + (1-eta2d)[:,None]*v1 +
                      (1+eta2d)[:,None]*v2 - (1+eta2d)[:,None]*v3)
    dPeta = 0.25 * (-(1-xi2d)[:,None]*v0  - (1+xi2d)[:,None]*v1  +
                      (1+xi2d)[:,None]*v2  + (1-xi2d)[:,None]*v3)
    J = np.linalg.norm(np.cross(dPxi, dPeta), axis=-1)
    return Q, w2d * J


# ---------------------------------------------------------------------------
# AIC builders
# ---------------------------------------------------------------------------

def build_doublet_aic(mesh: Mesh, include_image: bool = True) -> np.ndarray:
    """Doublet potential influence matrix L.

    L[i, j] = Ω(centroid_i, panel_j) / (4π),
    where Ω is the signed solid angle.  L[i, i] = 0 (self-term handled by
    the 0.5 diagonal in the BIE).

    Parameters
    ----------
    mesh         : Mesh
    include_image: if True, add mirror-image (y → −y) contribution.

    Returns
    -------
    L : (Np, Np) float
    """
    Np = mesh.n_panels
    pv = mesh.vertices[mesh.panels]   # (Np, 4, 3)
    cp = mesh.centroids               # (Np, 3)

    L = np.zeros((Np, Np))

    for j in range(Np):
        verts = pv[j]   # (4, 3)
        omega = _solid_angle_quad(cp, verts)
        L[:, j] += omega / (4.0 * np.pi)

        if include_image:
            # y → −y reflects the panel and reverses the vertex winding, so the
            # solid angle of the reflected panel has the OPPOSITE sign from what
            # the correct image should contribute.  Subtract to compensate.
            verts_img = verts.copy(); verts_img[:, 1] = -verts_img[:, 1]
            omega_img = _solid_angle_quad(cp, verts_img)
            L[:, j] -= omega_img / (4.0 * np.pi)

    np.fill_diagonal(L, 0.0)   # self-term → 0.5·I handled in solver
    return L


def build_source_potential_aic(
    mesh: Mesh,
    include_image: bool = True,
    n_gauss: int = 3,
) -> np.ndarray:
    """Source potential influence matrix S.

    S[i, j] = (1/4π) ∫_j (1/r) dS_j (Gauss quadrature).

    Parameters
    ----------
    mesh         : Mesh
    include_image: mirror-image source contribution.
    n_gauss      : Gauss points per direction (2, 3, or 4).

    Returns
    -------
    S : (Np, Np) float
    """
    Np = mesh.n_panels
    pv = mesh.vertices[mesh.panels]
    cp = mesh.centroids

    S = np.zeros((Np, Np))

    for j in range(Np):
        Q,   wJ   = _gauss_sample_quad(pv[j], n_gauss)           # (ng², 3), (ng²,)
        pv_img    = pv[j].copy(); pv_img[:, 1] = -pv_img[:, 1]
        Q_img, wJ_img = _gauss_sample_quad(pv_img, n_gauss)

        for i in range(Np):
            P = cp[i]
            r = np.linalg.norm(P[None, :] - Q, axis=-1)          # (ng²,)
            r = np.maximum(r, 1e-10)
            S[i, j] += np.sum(wJ / r)

            if include_image:
                r_img = np.linalg.norm(P[None, :] - Q_img, axis=-1)
                r_img = np.maximum(r_img, 1e-10)
                S[i, j] += np.sum(wJ_img / r_img)

    S /= 4.0 * np.pi
    return S


# ---------------------------------------------------------------------------
# Wake Kutta condition
# ---------------------------------------------------------------------------

def _build_wake_kutta(
    mesh: Mesh,
    wake_dir: np.ndarray,
    wake_length: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wake doublet AIC for Kutta condition embedding (right wing + image).

    For each TE spanwise strip k, approximates the semi-infinite doublet wake
    as a long finite quad panel extending ``wake_length`` units downstream in
    ``wake_dir``.  The image system (y → −y) is included for the symmetric
    full-wing model.

    Parameters
    ----------
    mesh        : Mesh (upper + lower panels, open tip).
    wake_dir    : (3,) unit-vector for wake direction (≈ V_∞/|V_∞|).
    wake_length : downstream extent; should be ≫ chord (default 100 × unit).

    Returns
    -------
    L_wake : (Np, Nte) solid-angle / (4π) influence of wake strip k at centroid i.
    A_w    : (Nte, 3) inner (root-side) TE midpoints.
    B_w    : (Nte, 3) outer (tip-side) TE midpoints.
    """
    te_u = mesh.te_pairs[:, 0]
    te_l = mesh.te_pairs[:, 1]
    Nte  = len(te_u)
    Np   = mesh.n_panels
    cp   = mesh.centroids           # (Np, 3)

    pu = mesh.vertices[mesh.panels[te_u]]   # (Nte, 4, 3) upper TE panels
    pl = mesh.vertices[mesh.panels[te_l]]   # (Nte, 4, 3) lower TE panels
    # Upper winding: v1=aft-inner, v2=aft-outer
    # Lower winding: v2=aft-outer, v3=aft-inner  (reversed winding on lower)
    A_w = 0.5 * (pu[:, 1, :] + pl[:, 3, :])   # inner TE midpoint  (Nte, 3)
    B_w = 0.5 * (pu[:, 2, :] + pl[:, 2, :])   # outer TE midpoint  (Nte, 3)

    d = wake_dir / np.linalg.norm(wake_dir)

    L_wake = np.zeros((Np, Nte))

    for k in range(Nte):
        A = A_w[k]; B = B_w[k]
        # Wake quad: CCW from above (normal ≈ +z, matching upper surface)
        # v0=inner-TE, v1=inner-far, v2=outer-far, v3=outer-TE
        wv = np.array([A,
                       A + wake_length * d,
                       B + wake_length * d,
                       B])

        omega = _solid_angle_quad(cp, wv)
        L_wake[:, k] += omega / (4.0 * np.pi)

        # Image wake (y → −y) reverses the vertex winding, so subtract the
        # solid angle of the reflected vertices (same fix as build_doublet_aic).
        wv_img = wv.copy()
        wv_img[:, 1] = -wv_img[:, 1]
        omega_img = _solid_angle_quad(cp, wv_img)
        L_wake[:, k] -= omega_img / (4.0 * np.pi)

    return L_wake, A_w, B_w


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_morino(
    mesh: Mesh,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    V_mag: float = 1.0,
    rho: float = 1.225,
    S_ref: float | None = None,
    b_ref: float | None = None,
    c_ref: float | None = None,
    x_ref: float | None = None,
    n_gauss: int = 2,
) -> dict:
    """Thick-surface aerodynamics: Hess-Smith thickness + VLM lifting effect.

    Combines two complementary solves:

    (A) **Hess-Smith source solve** on the actual thick surface → thickness Cp.
        Solves AIC_σ · σ = −(V_∞ · n̂) for the source strengths that enforce
        zero normal velocity due to thickness alone.  The tangential surface
        velocity is extracted and Cp_thickness computed separately for the upper
        and lower surface panels.  The self-panel normal singularity is removed
        by projecting out the normal component before computing |V_t|².

    (B) **VLM solve** on the mean camber surface → Kutta-Joukowski lift and
        the panel pressure difference ΔCp_j = 2Γ_j / (V_∞ · Δx_j) per panel.

    The two effects are superimposed:
        Cp_upper[j] = Cp_thickness_upper[j] − ΔCp_j / 2  (suction)
        Cp_lower[j] = Cp_thickness_lower[j] + ΔCp_j / 2  (pressure)

    Primary forces (CL, CDi, Cm) come from the VLM K-J integration — the most
    accurate estimate for a thin-wing VLM configuration.  The Cp-integrated
    forces are returned as a cross-check.

    Parameters
    ----------
    mesh      : Mesh from ``make_wing_mesh`` (upper + lower panels, open tip).
    alpha_deg : Angle of attack in degrees.
    beta_deg  : Sideslip in degrees.
    V_mag     : Freestream speed.
    rho       : Air density (kg/m³).
    S_ref     : Reference area (upper-surface panel area if None).
    b_ref     : Reference span (2 · max y if None).
    c_ref     : Reference chord (S_ref / (0.5 · b_ref) if None).
    x_ref     : Moment reference x (0.25 · c_ref if None).
    n_gauss   : Unused; kept for API compatibility.

    Returns
    -------
    dict
        CL, CDi, Cm, L, Di, M  — primary forces from VLM K-J.
        CL_cp, CDi_cp, Cm_cp   — secondary forces from Cp integration.
        Cp (Np,), V_surface (Np, 3), sigma (Np,)
        Gamma (Np_cam,), A (Np_cam, 3), B (Np_cam, 3)
        V_inf (3,), wake_dir (3,), S_ref, b_ref, c_ref.
    """
    from .solve import solve_vlm

    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)
    V_inf = V_mag * np.array([
        np.cos(alpha) * np.cos(beta),
        -np.sin(beta),
        np.sin(alpha),
    ])
    wake_dir = V_inf / np.linalg.norm(V_inf)

    if S_ref is None:
        S_ref = float(mesh.areas[mesh.surface_id == 0].sum())
    if b_ref is None:
        b_ref = 2.0 * float(mesh.vertices[:, 1].max())
    if c_ref is None:
        c_ref = S_ref / (0.5 * b_ref)
    if x_ref is None:
        x_ref = 0.25 * c_ref

    # ------------------------------------------------------------------ #
    # 1. Mean camber surface from thick mesh (vertex average)             #
    # ------------------------------------------------------------------ #
    n_up   = int((mesh.surface_id == 0).sum())
    off_lo = int(np.min(mesh.panels[mesh.surface_id == 1]))

    cam_vertices = 0.5 * (mesh.vertices[:off_lo] + mesh.vertices[off_lo:2 * off_lo])
    cam_panels   = mesh.panels[mesh.surface_id == 0].copy()   # refs [0, off_lo)
    te_u_idx     = mesh.te_pairs[:, 0]
    cam_te_pairs = np.column_stack([te_u_idx, te_u_idx])

    cam_mesh = Mesh(
        vertices=cam_vertices,
        panels=cam_panels,
        te_pairs=cam_te_pairs,
        wake_seed=mesh.wake_seed,
        surface_id=np.zeros(n_up, dtype=int),
    )

    # ------------------------------------------------------------------ #
    # 2. VLM solve on camber mesh → Γ, K-J forces, collocation points    #
    # ------------------------------------------------------------------ #
    vlm = solve_vlm(
        cam_mesh,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        V_mag=V_mag,
        rho=rho,
        S_ref=S_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        x_ref=x_ref,
    )

    # ------------------------------------------------------------------ #
    # 3. Hess-Smith source solve on thick surface → thickness Cp          #
    #                                                                     #
    # We solve at α = 0° to get the pure-thickness symmetric Cp.         #
    # Solving at the actual α would embed the stagnation-point shift      #
    # (a lifting effect) into σ, which then double-counts the VLM ΔCp.  #
    # Linear superposition: Cp = Cp_thickness(α=0) + Cp_lifting(α).     #
    # ------------------------------------------------------------------ #
    from .source_aic import build_source_aic, source_velocity_field

    V_inf_0  = V_mag * np.array([1.0, 0.0, 0.0])       # freestream at α=0
    AIC_src  = build_source_aic(mesh)
    rhs_src  = -(V_inf_0 @ mesh.normals.T)              # (Np,)
    sigma    = np.linalg.solve(AIC_src, rhs_src)        # (Np,)

    V_src    = source_velocity_field(mesh, sigma, mesh.centroids)  # (Np, 3)
    V_thk    = V_inf_0[None, :] + V_src                # (Np, 3)  α=0 total velocity
    V_n_thk  = np.einsum("ij,ij->i", V_thk, mesh.normals)         # (Np,) normal component
    Vt_sq    = np.einsum("ij,ij->i", V_thk, V_thk) - V_n_thk**2  # tangential speed²
    Cp_thickness = 1.0 - Vt_sq / V_mag**2              # (Np,) symmetric thickness Cp

    # ------------------------------------------------------------------ #
    # 4. Lifting pressure difference from K-J                             #
    #                                                                     #
    # ΔCp_j = (p_lower − p_upper)/q = 2Γ_j / (V_∞ · Δx_j)              #
    # where Δx_j ≈ panel area / spanwise bound-vortex length |B−A|.      #
    # ------------------------------------------------------------------ #
    span_len  = np.maximum(np.linalg.norm(vlm["B"] - vlm["A"], axis=-1), 1e-12)
    chord_len = cam_mesh.areas / span_len               # (Np_cam,) chord widths
    dCp       = 2.0 * vlm["Gamma"] / (V_mag * chord_len)  # (Np_cam,) pressure diff

    # ------------------------------------------------------------------ #
    # 5. Upper/lower Cp: superimpose thickness Cp and lifting ΔCp         #
    #                                                                     #
    # Camber panel j maps to: upper panel j, lower panel j + n_up.       #
    # (Same ordered layout from make_wing_mesh and make_vlm_mesh.)        #
    # ------------------------------------------------------------------ #
    Cp = np.zeros(mesh.n_panels)
    Cp[:n_up]         = Cp_thickness[:n_up]         - 0.5 * dCp   # upper: suction
    Cp[n_up:2 * n_up] = Cp_thickness[n_up:2 * n_up] + 0.5 * dCp  # lower: pressure
    # VLM thin-airfoil LE singularity can push lower-surface Cp above 1 when
    # cosine-clustered panels make chord_len tiny.  Incompressible Bernoulli
    # (V ≥ 0) is a hard physical upper bound.
    np.clip(Cp, -np.inf, 1.0, out=Cp)

    # ------------------------------------------------------------------ #
    # 6. Secondary Cp-based force integration (cross-check)               #
    # ------------------------------------------------------------------ #
    cp_forces = forces_from_cp(mesh, Cp, V_inf, rho, S_ref, b_ref, c_ref, x_ref)

    return {
        # Primary aerodynamic coefficients (VLM K-J — accurate)
        "CL":   vlm["CL"],
        "CDi":  vlm["CDi"],
        "Cm":   vlm["Cm"],
        "L":    vlm["L"],
        "Di":   vlm["Di"],
        "M":    vlm["M"],
        # Secondary coefficients from Cp integration (cross-check)
        "CL_cp":  cp_forces["CL"],
        "CDi_cp": cp_forces["CDi"],
        "Cm_cp":  cp_forces["Cm"],
        # Field data
        "Cp":        Cp,
        "forces":    cp_forces["forces"],
        "V_surface": V_thk,              # (2*n_up, 3) α=0 thickness surface velocity
        "sigma":     sigma,              # (Np,) source strengths from H-S solve
        "Gamma":     vlm["Gamma"],
        "A":         vlm["A"],
        "B":         vlm["B"],
        "V_inf":     V_inf,
        "wake_dir":  wake_dir,
        "S_ref":     S_ref,
        "b_ref":     b_ref,
        "c_ref":     c_ref,
    }


def _cp_from_mu_grad(
    mesh: Mesh,
    mu: np.ndarray,
    V_inf: np.ndarray,
    V_mag: float,
) -> np.ndarray:
    """Compute panel Cp from the doublet distribution gradient.

    The exterior surface velocity tangential to the panel is:

        V_t ≈ V_∞·t̂_chord + Δμ/Δs_chord  (chordwise)
              V_∞·t̂_span  + Δμ/Δs_span   (spanwise)

    and Cp = 1 − |V_t|² / V_mag².

    Uses centred differences where two neighbours are available; forward
    or backward difference at the panel boundaries.

    Parameters
    ----------
    mesh  : Mesh (upper panels first, then lower).
    mu    : (Np,) doublet distribution.
    V_inf : (3,) freestream velocity.
    V_mag : freestream speed.

    Returns
    -------
    Cp : (Np,) pressure coefficient.
    """
    Np = mesh.n_panels
    # Determine upper / lower panel counts from surface_id
    n_up = int((mesh.surface_id == 0).sum())
    n_lo = int((mesh.surface_id == 1).sum())

    # Infer grid shape from te_pairs / surface structure
    Nte = len(mesh.te_pairs)   # = n_span
    n_span  = Nte
    n_chord = n_up // n_span   # panels per surface

    Cp = np.zeros(Np)

    for surf, idx0, nlo_start in [("upper", 0, None), ("lower", n_up, None)]:
        if surf == "upper":
            idx_slice = slice(0, n_up)
        else:
            idx_slice = slice(n_up, n_up + n_lo)

        mu_surf = mu[idx_slice].reshape(n_chord, n_span)   # (nc, ns)
        cp_surf = mesh.centroids[idx_slice].reshape(n_chord, n_span, 3)

        Cp_surf = np.zeros((n_chord, n_span))

        for ic in range(n_chord):
            for js in range(n_span):
                c = cp_surf[ic, js]
                mu_c = mu_surf[ic, js]

                # ----- chordwise tangential velocity ----- #
                if ic == 0:
                    # Forward difference
                    c_next = cp_surf[ic + 1, js]
                    mu_next = mu_surf[ic + 1, js]
                    dc = c_next - c
                    t_chord = dc / (np.linalg.norm(dc) + 1e-30)
                    dmu_ds_chord = (mu_next - mu_c) / (np.linalg.norm(dc) + 1e-30)
                elif ic == n_chord - 1:
                    # Backward difference
                    c_prev = cp_surf[ic - 1, js]
                    mu_prev = mu_surf[ic - 1, js]
                    dc = c - c_prev
                    t_chord = dc / (np.linalg.norm(dc) + 1e-30)
                    dmu_ds_chord = (mu_c - mu_prev) / (np.linalg.norm(dc) + 1e-30)
                else:
                    # Centred difference
                    c_prev = cp_surf[ic - 1, js]
                    c_next = cp_surf[ic + 1, js]
                    mu_prev = mu_surf[ic - 1, js]
                    mu_next = mu_surf[ic + 1, js]
                    dc = c_next - c_prev
                    t_chord = dc / (np.linalg.norm(dc) + 1e-30)
                    dmu_ds_chord = (mu_next - mu_prev) / (np.linalg.norm(dc) + 1e-30)

                Vinf_t_chord = float(V_inf @ t_chord)
                V_t_chord = Vinf_t_chord + dmu_ds_chord

                # ----- spanwise tangential velocity ----- #
                if js == 0:
                    # Forward difference (root boundary)
                    c_next = cp_surf[ic, js + 1]
                    mu_next = mu_surf[ic, js + 1]
                    dc2 = c_next - c
                    t_span = dc2 / (np.linalg.norm(dc2) + 1e-30)
                    dmu_ds_span = (mu_next - mu_c) / (np.linalg.norm(dc2) + 1e-30)
                elif js == n_span - 1:
                    # Backward difference (tip boundary)
                    c_prev = cp_surf[ic, js - 1]
                    mu_prev = mu_surf[ic, js - 1]
                    dc2 = c - c_prev
                    t_span = dc2 / (np.linalg.norm(dc2) + 1e-30)
                    dmu_ds_span = (mu_c - mu_prev) / (np.linalg.norm(dc2) + 1e-30)
                else:
                    c_prev = cp_surf[ic, js - 1]
                    c_next = cp_surf[ic, js + 1]
                    mu_prev = mu_surf[ic, js - 1]
                    mu_next = mu_surf[ic, js + 1]
                    dc2 = c_next - c_prev
                    t_span = dc2 / (np.linalg.norm(dc2) + 1e-30)
                    dmu_ds_span = (mu_next - mu_prev) / (np.linalg.norm(dc2) + 1e-30)

                Vinf_t_span = float(V_inf @ t_span)
                V_t_span = Vinf_t_span + dmu_ds_span

                Cp_surf[ic, js] = 1.0 - (V_t_chord**2 + V_t_span**2) / V_mag**2

        if surf == "upper":
            Cp[:n_up] = Cp_surf.ravel()
        else:
            Cp[n_up:n_up + n_lo] = Cp_surf.ravel()

    return Cp
