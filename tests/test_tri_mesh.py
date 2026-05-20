"""Tests for triangulated-mesh support (Stage 7).

Covers:
- triangulate_mesh geometry: panel count, degenerate-quad area/normal/centroid
- te_pairs / te_verts topology preserved after triangulation
- solve_vlm: quad vs tri, rectangular AR=8, explicit S_ref → 0.000% error
- solve_morino: quad vs tri, NACA 0012 thick rectangular wing → 0.000% error
- Swept/tapered wing: error is bounded (parallelogram-completion approximation)

Key rule: always pass an explicit S_ref when using triangulated meshes.
The degenerate-quad area (first-diagonal triangle) is NOT the same as half
the original quad area for non-rectangular panels.
"""
from __future__ import annotations

import numpy as np
import pytest

from lspaero.geometry.mesh import Mesh
from lspaero.geometry.wing import make_vlm_mesh, make_wing_mesh
from lspaero.geometry.tri_utils import triangulate_mesh
from lspaero.solver.solve import solve_vlm
from lspaero.solver.morino import solve_morino


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rect_vlm_mesh(n_span=8, n_chord=1):
    """Rectangular AR=8 flat-plate VLM mesh (half_span=4, chord=1)."""
    return make_vlm_mesh(
        half_span=4.0, root_chord=1.0, tip_chord=1.0,
        n_span=n_span, n_chord=n_chord,
    )


def _rect_thick_mesh(n_span=6, n_chord=4):
    """Rectangular NACA-0012 thick mesh (half_span=4, chord=1), no tip cap."""
    return make_wing_mesh(
        half_span=4.0, root_chord=1.0, tip_chord=1.0,
        airfoil="0012", n_span=n_span, n_chord=n_chord,
        include_tip_cap=False,
    )


def _swept_vlm_mesh(n_span=6, n_chord=4):
    """Swept/tapered VLM mesh (half_span=5, root 2→tip 1, 30° LE sweep)."""
    return make_vlm_mesh(
        half_span=5.0, root_chord=2.0, tip_chord=1.0,
        sweep_le=30.0, n_span=n_span, n_chord=n_chord,
    )


# ---------------------------------------------------------------------------
# Geometry: triangulate_mesh basic invariants
# ---------------------------------------------------------------------------

def test_triangulate_panel_count_unchanged():
    """triangulate_mesh must not change the number of panels."""
    mesh_q = _rect_vlm_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    assert mesh_t.n_panels == mesh_q.n_panels


def test_triangulate_degenerate_quad_flag():
    """After triangulation every panel must satisfy panels[:,2] == panels[:,3]."""
    mesh_q = _rect_vlm_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    assert np.all(mesh_t.panels[:, 2] == mesh_t.panels[:, 3]), (
        "Not all panels are degenerate quads after triangulation"
    )


def test_triangulate_normals_unchanged_rectangular():
    """For rectangular panels, normals are identical after triangulation."""
    mesh_q = _rect_vlm_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    assert np.allclose(mesh_t.normals, mesh_q.normals, atol=1e-12), (
        "Normal vectors changed after triangulation of rectangular mesh"
    )


def test_triangulate_area_rectangular():
    """For rectangular panels, degenerate-quad area = half the quad area."""
    mesh_q = _rect_vlm_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    # Each rectangular panel is a parallelogram: first-diagonal triangle = half
    assert np.allclose(mesh_t.areas, 0.5 * mesh_q.areas, rtol=1e-10), (
        "Degenerate-quad area is not half the quad area for rectangular panels"
    )


def test_triangulate_te_pairs_preserved():
    """te_pairs indices must be unchanged after triangulation."""
    mesh_q = _rect_thick_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    assert np.array_equal(mesh_t.te_pairs, mesh_q.te_pairs), (
        "te_pairs changed after triangulation"
    )


def test_triangulate_te_verts_set():
    """triangulate_mesh must set te_verts from the TE panel geometry."""
    mesh_q = _rect_thick_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    assert mesh_t.te_verts is not None, "te_verts not set after triangulation"
    assert mesh_t.te_verts.shape == (len(mesh_q.te_pairs), 2, 3), (
        f"te_verts shape wrong: {mesh_t.te_verts.shape}"
    )


def test_triangulate_surface_id_preserved():
    """surface_id array must be unchanged after triangulation."""
    mesh_q = _rect_thick_mesh()
    mesh_t = triangulate_mesh(mesh_q)
    assert np.array_equal(mesh_t.surface_id, mesh_q.surface_id), (
        "surface_id changed after triangulation"
    )


# ---------------------------------------------------------------------------
# Solver: quad vs tri — rectangular wing (exact match expected)
# ---------------------------------------------------------------------------

def test_vlm_quad_vs_tri_rectangular_exact():
    """solve_vlm on rectangular mesh: quad vs tri must agree to < 0.001%.

    For a rectangular (parallelogram) panel, parallelogram completion is exact,
    so the VLM bound-vortex endpoints A and B are identical between quad and
    first-diagonal-triangle representations.  CL must be numerically identical.

    S_ref is passed explicitly (quad mesh area) to avoid the halved-area trap.
    """
    mesh_q = _rect_vlm_mesh(n_span=8, n_chord=1)
    mesh_t = triangulate_mesh(mesh_q)
    S_ref = float(mesh_q.areas.sum())   # physical half-wing area

    for alpha in [0.0, 2.0, 4.0, 6.0, 8.0]:
        r_q = solve_vlm(mesh_q, alpha_deg=alpha, S_ref=S_ref)
        r_t = solve_vlm(mesh_t, alpha_deg=alpha, S_ref=S_ref)
        CL_q = r_q["CL"]
        CL_t = r_t["CL"]
        # At α=0 both CL=0, skip relative error check
        if abs(CL_q) > 1e-8:
            err_pct = 100.0 * abs(CL_t - CL_q) / abs(CL_q)
            assert err_pct < 0.001, (
                f"VLM CL mismatch at α={alpha}°: quad={CL_q:.8f}, "
                f"tri={CL_t:.8f}, err={err_pct:.4f}%"
            )
        else:
            assert abs(CL_t - CL_q) < 1e-12, (
                f"VLM CL not zero at α=0°: quad={CL_q}, tri={CL_t}"
            )


def test_morino_quad_vs_tri_rectangular_exact():
    """solve_morino on rectangular NACA 0012: quad vs tri must agree to < 0.001%.

    The Morino solver builds a VLM camber mesh from the thick mesh vertices.
    For rectangular panels, parallelogram completion is exact in _ring_points,
    so the camber-mesh VLM gives the same Gamma → same CL.
    The source AIC handles degenerate quads correctly (zero-length edge skipped).

    S_ref is taken from the quad mesh upper surface and passed explicitly.
    """
    mesh_q = _rect_thick_mesh(n_span=6, n_chord=4)
    mesh_t = triangulate_mesh(mesh_q)
    S_ref = float(mesh_q.areas[mesh_q.surface_id == 0].sum())

    for alpha in [0.0, 4.0, 8.0]:
        r_q = solve_morino(mesh_q, alpha_deg=alpha, S_ref=S_ref)
        r_t = solve_morino(mesh_t, alpha_deg=alpha, S_ref=S_ref)
        CL_q = r_q["CL"]
        CL_t = r_t["CL"]
        if abs(CL_q) > 1e-8:
            err_pct = 100.0 * abs(CL_t - CL_q) / abs(CL_q)
            assert err_pct < 0.001, (
                f"Morino CL mismatch at α={alpha}°: quad={CL_q:.8f}, "
                f"tri={CL_t:.8f}, err={err_pct:.4f}%"
            )


# ---------------------------------------------------------------------------
# Solver: swept/tapered — bounded error from parallelogram approximation
# ---------------------------------------------------------------------------

def test_vlm_quad_vs_tri_swept_bounded():
    """solve_vlm on swept/tapered mesh: error < 10% (parallelogram approximation).

    For non-rectangular (swept/tapered) panels, the parallelogram completion
    in _ring_points introduces an approximation error proportional to Δchord
    per panel.  For the 6-strip coarse mesh used here this is ~5–6%.
    This is the expected, bounded, documented behaviour — not a solver defect.
    The test guards against regression (error must not exceed 10%).
    """
    mesh_q = _swept_vlm_mesh(n_span=6, n_chord=4)
    mesh_t = triangulate_mesh(mesh_q)
    S_ref = float(mesh_q.areas.sum())

    r_q = solve_vlm(mesh_q, alpha_deg=4.0, S_ref=S_ref)
    r_t = solve_vlm(mesh_t, alpha_deg=4.0, S_ref=S_ref)
    CL_q = r_q["CL"]
    CL_t = r_t["CL"]
    err_pct = 100.0 * abs(CL_t - CL_q) / abs(CL_q)
    # Bound: error must be finite and < 10% for this coarse swept mesh
    assert err_pct < 10.0, (
        f"VLM swept/tapered error too large: {err_pct:.2f}% "
        f"(quad CL={CL_q:.4f}, tri CL={CL_t:.4f})"
    )
    # Also assert the result is physically reasonable (positive lift at α=4°)
    assert CL_t > 0.1, f"Triangulated swept mesh gives implausible CL={CL_t}"
