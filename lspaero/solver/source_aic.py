"""Constant-strength source panel aerodynamic influence coefficients.

Used in the source-doublet (Morino) formulation for the thick-surface panel
solver.  Each panel j carries a source strength σ_j; this module computes the
normal velocity induced at every collocation point i by a unit-strength source
on panel j.

The implementation follows Hess & Smith (1967) via Katz & Plotkin §10.2 for
planar constant-strength source panels.  For the self-panel (i == j) the exact
half-space result AIC[i,i] = 0.5 is used directly.

A mirror-image source distribution (y → −y) is included to model the full
symmetric wing, matching the image system in build_panel_aic.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §10.2.
"""
from __future__ import annotations

import numpy as np

from ..geometry.mesh import Mesh


def _hess_smith_vel(
    u_p: np.ndarray,
    v_p: np.ndarray,
    w_p: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hess-Smith velocity (u, v, w) / (4π) at points (u_p, v_p, w_p).

    All coordinates in the panel's local frame.  The panel lies in the w=0
    plane with vertices at (uu[k], vv[k]).

    Parameters
    ----------
    u_p, v_p, w_p : (N,) eval-point local coordinates
    uu, vv : (4,) vertex local u- and v-coordinates

    Returns
    -------
    vel_u, vel_v, vel_w : (N,) velocity components / (4π)
    """
    N = len(u_p)
    vel_u = np.zeros(N)
    vel_v = np.zeros(N)
    vel_w = np.zeros(N)

    for k in range(4):
        k1 = (k + 1) % 4
        x_k  = uu[k];  y_k  = vv[k]
        x_k1 = uu[k1]; y_k1 = vv[k1]

        dx = x_k1 - x_k
        dy = y_k1 - y_k
        d  = np.sqrt(dx * dx + dy * dy)
        if d < 1e-14:
            continue

        r_k  = np.sqrt((u_p - x_k )**2 + (v_p - y_k )**2 + w_p**2)
        r_k1 = np.sqrt((u_p - x_k1)**2 + (v_p - y_k1)**2 + w_p**2)
        r_k  = np.maximum(r_k,  1e-14)
        r_k1 = np.maximum(r_k1, 1e-14)

        # Logarithmic term (in-plane)
        numer = np.maximum(np.abs(r_k + r_k1 - d), 1e-30)
        denom = r_k + r_k1 + d
        F_k   = np.log(numer / denom)

        vel_u += (dy / d) * F_k
        vel_v -= (dx / d) * F_k

        # Arctan term (normal component) — K&P Eq. 10.24: w = −σ/(4π) Σ(...)
        e_k  = (u_p - x_k )**2 + w_p**2
        h_k  = (u_p - x_k ) * (v_p - y_k )
        e_k1 = (u_p - x_k1)**2 + w_p**2
        h_k1 = (u_p - x_k1) * (v_p - y_k1)

        abs_dx = np.abs(dx)
        if abs_dx > 1e-10 * d:
            m = dy / dx
            tan1 = m * e_k  - h_k
            tan2 = m * e_k1 - h_k1
        else:
            big = 1e15
            tan1 = big * np.sign(dy) * np.sign(e_k  + 1e-30)
            tan2 = big * np.sign(dy) * np.sign(e_k1 + 1e-30)

        abs_w = np.abs(w_p)
        atan1 = np.arctan2(tan1, abs_w * r_k  + 1e-30)
        atan2 = np.arctan2(tan2, abs_w * r_k1 + 1e-30)
        vel_w -= np.sign(w_p + 1e-30 * (w_p == 0)) * (atan1 - atan2)

    return vel_u / (4.0 * np.pi), vel_v / (4.0 * np.pi), vel_w / (4.0 * np.pi)


def build_source_aic(mesh: Mesh) -> np.ndarray:
    """Build the source panel AIC matrix (right wing + mirror image).

    AIC_σ[i, j] = normal velocity at collocation i from a unit-strength source
    on panel j (right wing) PLUS the image source on the mirror panel j′
    (y → −y), matching the symmetry treatment in build_panel_aic.

    The diagonal is set to the exact half-space value 0.5; the image of the
    self-panel is far away and contributes only off-diagonal terms.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh (upper + lower panels; tip cap optional).

    Returns
    -------
    AIC_sigma : (Np, Np) float
    """
    Np  = mesh.n_panels
    pv  = mesh.vertices[mesh.panels]   # (Np, 4, 3)
    cp  = mesh.centroids               # (Np, 3)
    n   = mesh.normals                 # (Np, 3)

    AIC = np.zeros((Np, Np))

    # --- Local frames for all source panels (right wing) --- #
    e_x = pv[:, 1, :] - pv[:, 0, :]
    e_x /= np.linalg.norm(e_x, axis=-1, keepdims=True)
    e_z = n.copy()
    e_y = np.cross(e_z, e_x)
    e_y /= np.linalg.norm(e_y, axis=-1, keepdims=True)

    c_j = mesh.centroids                        # (Np, 3)
    dv  = pv - c_j[:, None, :]                 # (Np, 4, 3)
    u_v = np.einsum("pvi,pi->pv", dv, e_x)     # (Np, 4)
    v_v = np.einsum("pvi,pi->pv", dv, e_y)     # (Np, 4)

    # --- Image local frames (y → −y) --- #
    # e_x_img = (e_x_x, -e_x_y, e_x_z),  e_z_img = (n_x, -n_y, n_z)
    e_x_img = e_x.copy(); e_x_img[:, 1] = -e_x_img[:, 1]
    e_z_img = e_z.copy(); e_z_img[:, 1] = -e_z_img[:, 1]
    e_y_img = np.cross(e_z_img, e_x_img)
    e_y_img /= np.maximum(np.linalg.norm(e_y_img, axis=-1, keepdims=True), 1e-30)

    # Image centroid (just for offset computation)
    c_j_img = c_j.copy(); c_j_img[:, 1] = -c_j_img[:, 1]  # (Np, 3)

    # Image vertex local coords: same uu, vv as right wing (proven by symmetry)
    # (The image vertices in their own local frame have the same u,v coords)

    for j in range(Np):
        uu = u_v[j]   # (4,)
        vv = v_v[j]   # (4,)

        # ---- Right-wing source ---- #
        dr  = cp - c_j[j]           # (Np, 3)
        u_p = dr @ e_x[j]
        v_p = dr @ e_y[j]
        w_p = dr @ e_z[j]

        vel_u, vel_v, vel_w = _hess_smith_vel(u_p, v_p, w_p, uu, vv)
        vel_global = (vel_u[:, None] * e_x[j]
                      + vel_v[:, None] * e_y[j]
                      + vel_w[:, None] * e_z[j])
        AIC[:, j] += np.einsum("ij,ij->i", vel_global, n)

        # ---- Image source (y → −y) ---- #
        dr_img  = cp - c_j_img[j]   # (Np, 3) offset from image centroid
        u_img   = dr_img @ e_x_img[j]
        v_img   = dr_img @ e_y_img[j]
        w_img   = dr_img @ e_z_img[j]

        vel_u_i, vel_v_i, vel_w_i = _hess_smith_vel(u_img, v_img, w_img, uu, vv)
        vel_global_i = (vel_u_i[:, None] * e_x_img[j]
                        + vel_v_i[:, None] * e_y_img[j]
                        + vel_w_i[:, None] * e_z_img[j])
        AIC[:, j] += np.einsum("ij,ij->i", vel_global_i, n)

    # Exact half-space self-term (the image of panel j does not coincide with
    # the collocation point of panel j, so no correction needed there)
    np.fill_diagonal(AIC, 0.5)
    return AIC


def source_velocity_field(
    mesh: Mesh,
    sigma: np.ndarray,
    eval_points: np.ndarray,
) -> np.ndarray:
    """3-D velocity at arbitrary evaluation points from a source distribution.

    Returns V(r) = Σ_j σ_j · V_source_j(r) where V_source_j is the Hess-Smith
    velocity field of the j-th panel (right wing + y→−y image).

    The self-panel singularity is not handled here — eval_points should be
    displaced from panel centroids (e.g. the actual surface centroids of a
    thick mesh when sigma lives on the camber surface, or vice-versa).

    Parameters
    ----------
    mesh : Mesh
        Source panel mesh (panels carry the source distribution).
    sigma : (Np,) float
        Source strength on each panel.
    eval_points : (Ne, 3) float
        Evaluation points in global coordinates.

    Returns
    -------
    vel : (Ne, 3) float
        Total velocity vector at each evaluation point.
    """
    Np = mesh.n_panels
    Ne = len(eval_points)
    pv = mesh.vertices[mesh.panels]   # (Np, 4, 3)
    n  = mesh.normals                 # (Np, 3)

    # Local frames for right-wing panels
    e_x = pv[:, 1, :] - pv[:, 0, :]
    e_x /= np.linalg.norm(e_x, axis=-1, keepdims=True)
    e_z = n.copy()
    e_y = np.cross(e_z, e_x)
    e_y /= np.linalg.norm(e_y, axis=-1, keepdims=True)

    c_j = mesh.centroids              # (Np, 3)
    dv  = pv - c_j[:, None, :]       # (Np, 4, 3) vertices relative to centroid
    u_v = np.einsum("pvi,pi->pv", dv, e_x)   # (Np, 4) local coords
    v_v = np.einsum("pvi,pi->pv", dv, e_y)

    # Image local frames (y → −y)
    e_x_img = e_x.copy(); e_x_img[:, 1] = -e_x_img[:, 1]
    e_z_img = e_z.copy(); e_z_img[:, 1] = -e_z_img[:, 1]
    e_y_img = np.cross(e_z_img, e_x_img)
    e_y_img /= np.maximum(np.linalg.norm(e_y_img, axis=-1, keepdims=True), 1e-30)

    c_j_img = c_j.copy(); c_j_img[:, 1] = -c_j_img[:, 1]

    vel = np.zeros((Ne, 3))

    ep = eval_points   # (Ne, 3)

    for j in range(Np):
        if abs(sigma[j]) < 1e-30:
            continue
        uu = u_v[j]   # (4,)
        vv = v_v[j]   # (4,)

        # Right-wing contribution
        dr  = ep - c_j[j]
        u_p = dr @ e_x[j]
        v_p = dr @ e_y[j]
        w_p = dr @ e_z[j]
        vu, vv_, vw = _hess_smith_vel(u_p, v_p, w_p, uu, vv)
        vel_g = vu[:, None] * e_x[j] + vv_[:, None] * e_y[j] + vw[:, None] * e_z[j]
        vel += sigma[j] * vel_g

        # Image contribution (y → −y)
        dr_img = ep - c_j_img[j]
        u_p_i  = dr_img @ e_x_img[j]
        v_p_i  = dr_img @ e_y_img[j]
        w_p_i  = dr_img @ e_z_img[j]
        vu_i, vv_i, vw_i = _hess_smith_vel(u_p_i, v_p_i, w_p_i, uu, vv)
        vel_g_i = (vu_i[:, None] * e_x_img[j]
                   + vv_i[:, None] * e_y_img[j]
                   + vw_i[:, None] * e_z_img[j])
        vel += sigma[j] * vel_g_i

    return vel
