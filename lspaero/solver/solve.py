"""VLM solver driver.

Solves the linear system A·Γ = b for panel circulations, then computes
near-field aerodynamic forces via Kutta-Joukowski.

Typical usage::

    from lspaero.geometry.wing import make_vlm_mesh
    from lspaero.solver.solve import solve_vlm

    mesh = make_vlm_mesh(half_span=5, root_chord=2, n_span=20, n_chord=8)
    result = solve_vlm(mesh, alpha_deg=5.0)
    print(result["CL"], result["CDi"])
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..geometry.mesh import Mesh
from ..physics.forces import compute_forces
from .influence import build_aic, induced_velocity_at


def solve_vlm(
    mesh: Mesh,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    V_mag: float = 1.0,
    rho: float = 1.225,
    mach: float = 0.0,
    S_ref: float | None = None,
    b_ref: float | None = None,
    c_ref: float | None = None,
    x_ref: float | None = None,
) -> dict:
    """Solve the VLM for given flow conditions on a camber-surface mesh.

    The right half-wing mesh is solved with a symmetric image system so that
    results correspond to a full symmetric wing.

    Prandtl-Glauert correction is applied when mach > 0: the mesh is stretched
    chordwise by 1/β (x' = x/β), the incompressible solve is run on the
    transformed geometry, and the resulting K-J forces already equal the
    PG-corrected physical forces when normalised by the physical S_ref.

    Parameters
    ----------
    mesh : Mesh
        Camber-surface mesh from ``make_vlm_mesh``.
    alpha_deg : float
        Angle of attack in degrees (pitch up positive).
    beta_deg : float
        Side-slip angle in degrees (nose-left positive).  Breaks symmetry;
        image system is kept but Γ will be asymmetric.
    V_mag : float
        Freestream speed (m/s or non-dimensional, consistent with rho).
    rho : float
        Air density (kg/m³).
    mach : float
        Freestream Mach number (0 ≤ M < 1).  0 = incompressible (default).
    S_ref : float, optional
        Reference area.  Defaults to sum of half-wing panel areas of the
        **original** (physical) mesh.
    b_ref : float, optional
        Reference span.  Defaults to 2 × max(y) of the original mesh.
    c_ref : float, optional
        Reference chord (MAC).  Defaults to S_ref / (0.5 × b_ref).
    x_ref : float, optional
        Moment reference x-coordinate.  Defaults to 0.25 × c_ref.

    Returns
    -------
    dict
        CL, CDi, Cm, Gamma (Np,), forces (Np,3), A (Np,3), B (Np,3),
        cp (Np,3), V_inf (3,), wake_dir (3,), mach, beta.
    """
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)

    # Freestream velocity in body frame (x=chordwise, y=spanwise, z=up)
    V_inf = V_mag * np.array([
        np.cos(alpha) * np.cos(beta),
        -np.sin(beta),
        np.sin(alpha),
    ])

    # Fixed planar wake: aligned with freestream direction
    wake_dir = V_inf / np.linalg.norm(V_inf)

    # Reference quantities from the PHYSICAL (pre-PG) mesh
    if S_ref is None:
        S_ref = float(mesh.areas.sum())
    if b_ref is None:
        b_ref = 2.0 * float(mesh.vertices[:, 1].max())
    if c_ref is None:
        c_ref = S_ref / (0.5 * b_ref)
    if x_ref is None:
        x_ref = 0.25 * c_ref

    # Prandtl-Glauert: stretch mesh in x by 1/β before the incompressible solve.
    # Reference areas/span/chord stay as physical values; only the moment
    # reference x-coordinate must be scaled to match the stretched coordinates.
    pg = 1.0   # β factor; updated below when mach > 0
    solve_mesh = mesh
    x_ref_solve = x_ref

    if mach > 1e-6:
        from ..physics.pg_correction import pg_stretch_mesh
        solve_mesh, pg = pg_stretch_mesh(mesh, mach)
        x_ref_solve = x_ref / pg   # moment reference in stretched coordinates

    # Build AIC and get vortex-ring geometry
    AIC, A, B, cp = build_aic(solve_mesh, wake_dir)

    # RHS: normal wash from freestream (no-penetration BC)
    rhs = -(V_inf @ solve_mesh.normals.T)   # (Np,)

    # Solve for circulations
    lu, piv = lu_factor(AIC)
    Gamma = lu_solve((lu, piv), rhs)   # (Np,)

    # Induced velocity at each bound-vortex midpoint for K-J forces
    mid_pts = 0.5 * (A + B)    # (Np, 3)
    V_ind = induced_velocity_at(mid_pts, A, B, Gamma, wake_dir)

    # K-J forces on the stretched geometry with physical S_ref give PG-corrected
    # aerodynamic coefficients directly (CL = CL_incomp / β, etc.).
    result = compute_forces(
        A, B, Gamma, V_inf, V_ind, rho, S_ref, b_ref, c_ref, x_ref_solve
    )
    result.update({
        "Gamma": Gamma,
        "A": A,
        "B": B,
        "cp": cp,
        "V_inf": V_inf,
        "wake_dir": wake_dir,
        "S_ref": S_ref,
        "b_ref": b_ref,
        "c_ref": c_ref,
        "mach": mach,
        "beta": pg,
    })
    return result
