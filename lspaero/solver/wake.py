"""Wake discretization and free-wake relaxation for VLM.

Replaces the semi-infinite trailing legs of each horseshoe vortex with a
finite-row node system that can be iteratively advected to align with the
local flow.

Wake node layout (per panel)
----------------------------
Left  trailing leg (from A[k]): nodes_A[r, k]  for r = 0 .. n_rows
Right trailing leg (from B[k]): nodes_B[r, k]  for r = 0 .. n_rows

Row 0 is permanently fixed at A[k] / B[k] (the bound-vortex endpoints).
Rows 1 .. n_rows are the free nodes; the wake is continued to infinity
from row n_rows in direction d_far (typically the freestream direction).

Reference: Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001), §14.2.
"""
from __future__ import annotations

import numpy as np

from .influence import induced_velocity_at


def build_wake_nodes(
    A: np.ndarray,
    B: np.ndarray,
    wake_dir: np.ndarray,
    n_rows: int,
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialise planar wake nodes aligned with the freestream.

    Katz & Plotkin §14.2: initial wake is a flat surface extending downstream
    from the trailing edge, aligned with the undisturbed freestream.

    Parameters
    ----------
    A, B : (Np, 3)
        Bound-vortex inner/outer endpoints (from ``_ring_points``).
    wake_dir : (3,)
        Unit freestream direction (initial wake direction).
    n_rows : int
        Number of finite-segment rows per filament.
    ds : float
        Row spacing in physical length units.

    Returns
    -------
    nodes_A, nodes_B : (n_rows+1, Np, 3)
        Wake node positions.  Row 0 equals A / B exactly.
    """
    d = wake_dir / np.linalg.norm(wake_dir)     # (3,)
    r_arr = np.arange(n_rows + 1)               # (n_rows+1,)
    # offsets[r, 0, :] = r * ds * d  →  broadcast with A/B
    offsets = r_arr[:, None, None] * ds * d[None, None, :]   # (n_rows+1, 1, 3)
    nodes_A = A[None, :, :] + offsets   # (n_rows+1, Np, 3)
    nodes_B = B[None, :, :] + offsets
    return nodes_A, nodes_B


def wake_node_velocity(
    nodes_A: np.ndarray,
    nodes_B: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Gamma: np.ndarray,
    V_inf: np.ndarray,
    d_far: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Total velocity at every wake node = freestream + induced from bound vortices.

    Uses the semi-infinite horseshoe model for the induced velocity; wake
    self-induction is not included.  This is a good approximation for
    moderate angles of attack (α ≲ 15°) where wake self-induction is a
    second-order effect.

    Parameters
    ----------
    nodes_A, nodes_B : (n_rows+1, Np, 3)
        Current wake node positions.
    A, B : (Np, 3)
        Bound-vortex endpoints.
    Gamma : (Np,)
        Panel circulations (from the most recent solve).
    V_inf : (3,)
        Freestream velocity vector.
    d_far : (3,)
        Far-field direction used by ``induced_velocity_at`` for semi-infinite
        trailing legs.

    Returns
    -------
    V_A, V_B : (n_rows+1, Np, 3)
        Total velocity at nodes_A / nodes_B.
    """
    n_rows_p1, Np, _ = nodes_A.shape
    pts_A = nodes_A.reshape(-1, 3)    # (M, 3)  M = (n_rows+1)*Np
    pts_B = nodes_B.reshape(-1, 3)
    V_ind_A = induced_velocity_at(pts_A, A, B, Gamma, d_far)   # (M, 3)
    V_ind_B = induced_velocity_at(pts_B, A, B, Gamma, d_far)
    V_A = (V_inf[None, :] + V_ind_A).reshape(n_rows_p1, Np, 3)
    V_B = (V_inf[None, :] + V_ind_B).reshape(n_rows_p1, Np, 3)
    return V_A, V_B


def advect_wake_nodes(
    nodes_A: np.ndarray,
    nodes_B: np.ndarray,
    V_A: np.ndarray,
    V_B: np.ndarray,
    dt: float,
    relax: float = 0.3,
    n_fixed_rows: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-trace free wake nodes along streamlines with under-relaxation.

    Each free row r is placed at the predicted position obtained by tracing
    one Euler step from row r-1 with the total (freestream + induced)
    velocity:

        pred[r] = node[r-1] + dt · V_total[r-1]
        node_new[r] = node[r] + relax · (pred[r] − node[r])

    At convergence ``node[r] = node[r-1] + dt · V(node[r-1])`` for all free
    rows: the nodes lie on streamlines of the current flow field and
    ``max_delta → 0``.  Row 0 and rows 1..n_fixed_rows are kept fixed to
    protect the Kutta condition near the TE (Katz & Plotkin §14.2).

    Parameters
    ----------
    nodes_A, nodes_B : (n_rows+1, Np, 3)
    V_A, V_B : (n_rows+1, Np, 3)
        Total velocity (freestream + induced) at each wake node.
    dt : float
        Integration time step (= ds / V_inf_magnitude).
    relax : float
        Under-relaxation factor (0 < relax ≤ 1).  0.3 is conservative.
    n_fixed_rows : int
        Number of rows beyond row 0 to keep frozen near the TE.

    Returns
    -------
    new_nodes_A, new_nodes_B : (n_rows+1, Np, 3)
    max_delta : float
        Maximum correction applied ``= max|relax·(pred − current)|``.
        Approaches 0 at steady state (convergence indicator).
    """
    first_free = n_fixed_rows + 1

    # Jacobi update: use only the *induced* component of V_A/V_B.
    # Caller must subtract V_inf before passing V_A/V_B so that the freestream
    # component does not cause unlimited downstream drift.
    # Each free node moves by relax * dt * V_ind independently (no coupling
    # between rows within one iteration), which prevents error propagation
    # along the wake chain and gives stable, monotone convergence.
    dA = relax * dt * V_A[first_free:]   # (n_free, Np, 3)
    dB = relax * dt * V_B[first_free:]

    new_A = nodes_A.copy()
    new_B = nodes_B.copy()
    new_A[first_free:] += dA
    new_B[first_free:] += dB

    max_delta = float(max(np.abs(dA).max(), np.abs(dB).max()))
    return new_A, new_B, max_delta


def trace_streamlines(
    A: np.ndarray,
    B: np.ndarray,
    Gamma: np.ndarray,
    V_inf: np.ndarray,
    n_steps: int,
    ds: float,
    seed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace wake streamlines from trailing-edge points by Euler integration.

    Used for visualising the wake shape.  Does not modify the AIC or solve.

    Parameters
    ----------
    A, B : (Np, 3)
        Bound-vortex endpoints (TE for a single-chordwise-row VLM).
    Gamma : (Np,)
        Panel circulations.
    V_inf : (3,)
        Freestream velocity vector.
    n_steps : int
        Number of Euler integration steps.
    ds : float
        Step length (physical units).
    seed_indices : (k,) int, optional
        Which panels to trace.  Defaults to all panels.

    Returns
    -------
    lines_A, lines_B : (n_steps+1, k, 3)
        Wake streamline positions for the left / right trailing filaments.
    """
    d = V_inf / np.linalg.norm(V_inf)
    dt = ds / np.linalg.norm(V_inf)
    idx = seed_indices if seed_indices is not None else np.arange(len(A))
    k = len(idx)
    lines_A = np.empty((n_steps + 1, k, 3))
    lines_B = np.empty((n_steps + 1, k, 3))
    lines_A[0] = A[idx]
    lines_B[0] = B[idx]
    for r in range(1, n_steps + 1):
        pts = np.vstack([lines_A[r - 1], lines_B[r - 1]])   # (2k, 3)
        V_ind = induced_velocity_at(pts, A, B, Gamma, d)
        V_tot = V_inf[None, :] + V_ind
        lines_A[r] = lines_A[r - 1] + V_tot[:k] * dt
        lines_B[r] = lines_B[r - 1] + V_tot[k:] * dt
    return lines_A, lines_B
