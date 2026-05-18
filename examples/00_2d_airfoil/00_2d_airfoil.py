"""Stage 1 demo: 2D Hess-Smith panel method on a NACA 4-digit airfoil.

Method
------
Constant-source + single-vortex (Hess & Smith 1967).  Each panel carries an
unknown source strength σ_j; all panels share a single unknown vortex strength
γ (per unit length).  The Kutta condition closes the system.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed., §11.2.

Usage
-----
    python examples/00_2d_airfoil.py
Saves two PNG figures in examples/:
    stage1_cp.png   — Cp distributions at α = 0°, 4°, 8°
    stage1_cl.png   — CL vs α sweep, compared with 2π/rad thin-airfoil slope
"""
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from lspaero.geometry.naca import naca4


# ---------------------------------------------------------------------------
# Panel geometry helpers
# ---------------------------------------------------------------------------

def _panel_geometry(x: np.ndarray, z: np.ndarray):
    """Return collocation points, tangents, outward normals, and lengths.

    Parameters
    ----------
    x, z : (N+1,) ndarray
        Node coordinates in CCW order (lower TE → LE → upper TE).

    Returns
    -------
    xm, zm : (N,)  collocation points (panel midpoints)
    tx, tz : (N,)  unit tangent vectors
    nx, nz : (N,)  outward unit normals (left-hand of travel direction)
    l      : (N,)  panel lengths
    alpha  : (N,)  panel angle from +x axis [rad]
    """
    dx = np.diff(x)
    dz = np.diff(z)
    l = np.hypot(dx, dz)
    tx = dx / l
    tz = dz / l
    # Left-hand normal for CCW traversal points outward
    nx = -tz
    nz =  tx
    xm = 0.5 * (x[:-1] + x[1:])
    zm = 0.5 * (z[:-1] + z[1:])
    alpha = np.arctan2(dz, dx)
    return xm, zm, tx, tz, nx, nz, l, alpha


# ---------------------------------------------------------------------------
# Influence coefficient assembly  (fully vectorized, no loops over panels)
# ---------------------------------------------------------------------------

def _source_velocity(xm_i, zm_i, x_j0, z_j0, alpha_j, l_j):
    """Velocity (global x-z) at each collocation point from each unit source.

    Uses the closed-form 2D source-panel integral (Katz & Plotkin §10.4).

    Parameters
    ----------
    xm_i, zm_i : (N,)   collocation-point coordinates
    x_j0, z_j0 : (N,)   start-node coordinates of each source panel
    alpha_j     : (N,)   panel angle [rad]
    l_j         : (N,)   panel lengths

    Returns
    -------
    u_s, w_s : (N, N)  global velocity components; index [i, j].
    """
    # Relative displacement from panel-j start to collocation i  → (N, N)
    dx = xm_i[:, None] - x_j0[None, :]   # [i, j]
    dz = zm_i[:, None] - z_j0[None, :]

    cos_a = np.cos(alpha_j)[None, :]      # (1, N) broadcast over rows
    sin_a = np.sin(alpha_j)[None, :]

    # Local (along-panel, normal-to-panel) coordinates
    xi  =  dx * cos_a + dz * sin_a
    eta = -dx * sin_a + dz * cos_a

    r1_sq = xi**2 + eta**2
    r2_sq = (xi - l_j[None, :])**2 + eta**2

    # Regularise to avoid log(0) at degenerate points (shouldn't happen for
    # a valid mesh, but protects against floating-point edge cases)
    eps = 1e-14
    r1_sq = np.maximum(r1_sq, eps)
    r2_sq = np.maximum(r2_sq, eps)

    # Use the single atan2 identity  θ₂−θ₁ = atan2(η·l, ξ(ξ−l)+η²)
    # to avoid sign ambiguity when η→0 via two separate arctan2 calls.
    # This formula always yields +π for on-panel points (η=0, 0<ξ<l),
    # giving the physically correct outward-normal convention.
    numer = eta * l_j[None, :]
    denom = xi * (xi - l_j[None, :]) + eta**2

    inv2pi = 0.5 / np.pi
    uL = inv2pi * 0.5 * np.log(r1_sq / r2_sq)  # = (1/2π) ln(r1/r2)
    wL = inv2pi * np.arctan2(numer, denom)

    # Rotate local → global
    u_s = uL * cos_a - wL * sin_a
    w_s = uL * sin_a + wL * cos_a
    return u_s, w_s


def _build_influence_matrices(xm, zm, x_node, z_node, tx, tz, nx, nz, l, alpha):
    """Return A (normal-source) and D (tangential-source) matrices.

    Vortex contributions follow from the 90°-rotation identity:
        normal from vortex  = tangential from source  →  C = D
        tangential from vortex = -(normal from source) →  F = -A

    Parameters
    ----------
    All arrays shape (N,) except x_node, z_node which are (N+1,).

    Returns
    -------
    A, D : (N, N) ndarray
        A[i, j] = normal velocity at collocation i from unit-strength source j.
        D[i, j] = tangential velocity at collocation i from unit-strength source j.
    """
    x_j0 = x_node[:-1]
    z_j0 = z_node[:-1]

    u_s, w_s = _source_velocity(xm, zm, x_j0, z_j0, alpha, l)

    # Normal component: n̂_i · (u_s, w_s),  n̂_i = (nx[i], nz[i])
    A = u_s * nx[:, None] + w_s * nz[:, None]

    # Tangential component: t̂_i · (u_s, w_s),  t̂_i = (tx[i], tz[i])
    D = u_s * tx[:, None] + w_s * tz[:, None]

    # Self-panel (i == j): the source panel's own outward normal velocity is
    # analytically σ/2.  Floating-point sign of η (zero by construction) can
    # flip atan2 to −π, giving −0.5 instead of +0.5.  Enforce the exact value.
    np.fill_diagonal(A, 0.5)

    return A, D


# ---------------------------------------------------------------------------
# Hess-Smith solver
# ---------------------------------------------------------------------------

def solve_hess_smith(x: np.ndarray, z: np.ndarray, alpha_deg: float,
                     v_inf: float = 1.0):
    """Solve the 2D Hess-Smith panel problem.

    Parameters
    ----------
    x, z : (N+1,) ndarray
        Airfoil node coordinates (CCW, lower TE → LE → upper TE, unit chord).
    alpha_deg : float
        Angle of attack [degrees].
    v_inf : float
        Freestream speed [m/s] (default 1).

    Returns
    -------
    sigma : (N,) source strengths [m/s]
    gamma : float  vortex strength per unit length [m/s]
    Cp    : (N,) pressure coefficient at each panel collocation point
    CL    : float  lift coefficient (from Kutta-Joukowski)
    """
    alpha_rad = np.deg2rad(alpha_deg)
    Vx = v_inf * np.cos(alpha_rad)   # freestream x-component
    Vz = v_inf * np.sin(alpha_rad)   # freestream z-component

    xm, zm, tx, tz, nx, nz, l, panel_angle = _panel_geometry(x, z)
    N = len(l)

    A, D = _build_influence_matrices(xm, zm, x, z, tx, tz, nx, nz, l, panel_angle)

    # Vortex normal influence at each collocation point: B_i = Σ_j D[i,j]
    B = D.sum(axis=1)          # (N,)

    # -------------------------------------------------------------------
    # Assemble (N+1) × (N+1) system
    # Unknowns: [σ_0, ..., σ_{N-1}, γ]
    # -------------------------------------------------------------------
    M = np.empty((N + 1, N + 1))

    # Rows 0..N-1: no-penetration BC at each collocation point
    M[:N, :N] = A
    M[:N,  N] = B

    # Row N: Kutta condition  Vt[0] + Vt[N-1] = 0
    # Vortex tangential influence: E_i = Σ_j F[i,j] = -Σ_j A[i,j]
    E0   = -A[0,   :].sum()
    E_N1 = -A[N-1, :].sum()
    M[N, :N] = D[0, :] + D[N-1, :]
    M[N,  N] = E0 + E_N1

    # RHS
    rhs = np.empty(N + 1)
    rhs[:N] = -(Vx * nx + Vz * nz)
    rhs[N]  = -(Vx * (tx[0] + tx[N-1]) + Vz * (tz[0] + tz[N-1]))

    sol = np.linalg.solve(M, rhs)
    sigma = sol[:N]
    gamma = sol[N]

    # -------------------------------------------------------------------
    # Post-process: tangential velocity → Cp
    # -------------------------------------------------------------------
    Vt = D @ sigma + (-A.sum(axis=1)) * gamma + (Vx * tx + Vz * tz)
    Cp = 1.0 - (Vt / v_inf) ** 2

    # Kutta-Joukowski: L = ρ V∞ × Γ.  In 2D, positive CCW Γ gives downforce;
    # for positive lift (α>0) the vortex is CW (γ<0), so CL = −2Γ/(V∞ c).
    Gamma = gamma * l.sum()
    CL = -2.0 * Gamma / (v_inf * 1.0)   # chord c = 1

    return sigma, gamma, Cp, CL


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _split_upper_lower(xm, Cp, N):
    """Split collocation arrays into lower-surface and upper-surface parts.

    For N panels with CCW ordering (lower TE → LE → upper TE):
      lower: indices 0 .. N//2 - 1    (x decreasing, TE → LE)
      upper: indices N//2 .. N-1      (x increasing, LE → TE)
    """
    n_half = N // 2
    # Lower surface (reversed so x increases from LE to TE for plotting)
    xl = xm[:n_half][::-1]
    Cpl = Cp[:n_half][::-1]
    # Upper surface
    xu = xm[n_half:]
    Cpu = Cp[n_half:]
    return xl, Cpl, xu, Cpu


def plot_cp(results, out_path):
    """Plot Cp vs x/c for multiple angle-of-attack cases.

    Parameters
    ----------
    results : list of (alpha_deg, xm, Cp, N) tuples
    out_path : str
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (alpha_deg, xm, Cp, N) in zip(axes, results):
        xl, Cpl, xu, Cpu = _split_upper_lower(xm, Cp, N)
        ax.plot(xu, Cpu, "b-", lw=1.5, label="upper")
        ax.plot(xl, Cpl, "r-", lw=1.5, label="lower")
        ax.invert_yaxis()
        ax.set_xlabel("x/c")
        ax.set_title(f"α = {alpha_deg}°")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4)

    axes[0].set_ylabel("Cp  (inverted, suction up)")
    fig.suptitle("NACA 0012 — Hess-Smith (2D inviscid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_cl_alpha(alphas, CLs, out_path):
    """Plot CL vs α and overlay the thin-airfoil-theory line (2π/rad)."""
    alphas_rad = np.deg2rad(alphas)
    CL_theory = 2.0 * np.pi * alphas_rad

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(alphas, CLs, "ko-", ms=4, label="Hess-Smith (panel method)")
    ax.plot(alphas, CL_theory, "b--", lw=1.5, label=r"$2\pi\alpha$ (thin-airfoil theory)")
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("CL")
    ax.set_title("NACA 0012 — CL vs α")
    ax.legend()
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_PANELS = 100   # 50 per surface; cosine-clustered
    CODE = "0012"

    x, z = naca4(CODE, N_PANELS)
    xm, zm, tx, tz, nx, nz, l, panel_angle = _panel_geometry(x, z)
    N = len(l)

    out_dir = os.path.dirname(__file__)

    # --- Cp plots at selected angles of attack ---
    alpha_cp_cases = [0.0, 4.0, 8.0]
    results = []
    for alpha_deg in alpha_cp_cases:
        sigma, gamma, Cp, CL = solve_hess_smith(x, z, alpha_deg)
        results.append((alpha_deg, xm, Cp, N))
        print(f"  α = {alpha_deg:5.1f}°  CL = {CL:.4f}  "
              f"(2π·α = {2*np.pi*np.deg2rad(alpha_deg):.4f})")

    plot_cp(results, os.path.join(out_dir, "stage1_cp.png"))

    # --- CL vs α sweep ---
    alpha_sweep = np.linspace(-8.0, 16.0, 25)
    CLs = []
    for alpha_deg in alpha_sweep:
        _, _, _, CL = solve_hess_smith(x, z, alpha_deg)
        CLs.append(CL)
    CLs = np.array(CLs)

    plot_cl_alpha(alpha_sweep, CLs, os.path.join(out_dir, "stage1_cl.png"))

    # --- Verify CL slope ---
    # Fit a line over -4° to 8°
    mask = (alpha_sweep >= -4.0) & (alpha_sweep <= 8.0)
    slope_fit = np.polyfit(np.deg2rad(alpha_sweep[mask]), CLs[mask], 1)[0]
    print(f"\n  CL_alpha (panel method) = {slope_fit:.4f} /rad")
    print(f"  CL_alpha (thin-airfoil) = {2*np.pi:.4f} /rad")
    print(f"  Error = {abs(slope_fit - 2*np.pi)/( 2*np.pi)*100:.2f} %")
    print("\nStage 1 complete — figures written to examples/")
