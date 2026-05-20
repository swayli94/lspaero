"""Prandtl-Glauert compressibility correction.

Applies the PG similarity rule that maps compressible inviscid flow at Mach M
to an equivalent incompressible problem on a stretched geometry.

Transform: stretch every x-coordinate (freestream / chordwise direction) by
1/β where β = √(1 − M²).  The incompressible panel solve on the stretched mesh
then yields PG-corrected lift, drag, and moment coefficients directly via the
Kutta-Joukowski force formula.  Surface Cp requires an additional division by β.

Derivation sketch (Katz & Plotkin §12.3-12.5)
----------------------------------------------
The compressible small-disturbance equation
    β² φ_xx + φ_yy + φ_zz = 0
becomes Laplace's equation under the coordinate stretch x₁ = x/β.
Boundary conditions (no-penetration, Kutta) are unchanged.  The K-J force per
unit span is ρVΓ — identical in physical and stretched frames because span (y)
is untouched.  With the physical reference area S_ref held fixed:

    CL_pg  = L_1 / (q · S_ref) = CL_incomp / β      (from K-J, no extra scaling)
    Cp_pg  = Cp_1 / β                                 (explicit division needed)

where the subscript 1 denotes quantities on the stretched mesh.
"""
from __future__ import annotations

import numpy as np

from ..geometry.mesh import Mesh


def pg_beta(mach: float) -> float:
    """Return β = √(1 − M²) for M < 1.

    Parameters
    ----------
    mach : float  Freestream Mach number (must be < 1).

    Returns
    -------
    beta : float
    """
    if mach >= 1.0:
        raise ValueError(f"PG correction requires M < 1 (got M = {mach:.4f})")
    return float(np.sqrt(max(0.0, 1.0 - mach ** 2)))


def pg_stretch_mesh(mesh: Mesh, mach: float) -> tuple[Mesh, float]:
    """Return a chordwise-stretched mesh for the PG-transformed incompressible solve.

    Every vertex x-coordinate is scaled by 1/β; y and z are unchanged.
    The caller must compute S_ref, b_ref, c_ref from the **original** mesh
    before calling this function, then pass them unchanged into the solver.

    For the pitching moment, the moment reference x-coordinate must also be
    scaled to x_ref / β when it is passed to ``compute_forces`` / ``solve_vlm``,
    so that the moment arm is measured correctly in stretched coordinates.

    Parameters
    ----------
    mesh : Mesh   Physical wing mesh.
    mach : float  Freestream Mach number (0 ≤ M < 1).

    Returns
    -------
    mesh_pg : Mesh  Stretched mesh (x' = x/β).
    beta    : float  β = √(1 − M²).
    """
    beta = pg_beta(mach)
    if beta > 1.0 - 1e-12:   # incompressible limit — skip transform
        return mesh, beta

    v_pg = mesh.vertices.copy()
    v_pg[:, 0] /= beta   # x' = x / β

    wake_pg = mesh.wake_seed.copy() if mesh.wake_seed is not None else None
    if wake_pg is not None:
        wake_pg[:, 0] /= beta

    mesh_pg = Mesh(
        vertices=v_pg,
        panels=mesh.panels.copy(),
        te_pairs=mesh.te_pairs.copy(),
        wake_seed=wake_pg,
        surface_id=mesh.surface_id.copy(),
    )
    return mesh_pg, beta
