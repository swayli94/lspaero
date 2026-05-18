"""Tests for the Mesh data structure and the parametric wing generator.

Reference checks:
- Normals are unit vectors (trivial invariant).
- Upper-surface normals have +z component for a flat/unswept wing.
- Lower-surface normals have −z component.
- Tip-cap normals have +y component (right half-wing).
- TE pairs reference valid panel indices.
- All panel areas are positive.
- Panels are nearly planar (max non-planarity check).
"""
import numpy as np
import pytest

from lspaero.geometry.mesh import Mesh
from lspaero.geometry.wing import (
    SURF_LOWER,
    SURF_TIP,
    SURF_UPPER,
    make_wing_mesh,
)


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def rect_mesh():
    """Rectangular flat wing — no sweep, no taper, no dihedral, no twist."""
    return make_wing_mesh(
        half_span=5.0,
        root_chord=2.0,
        tip_chord=2.0,
        sweep_le=0.0,
        dihedral=0.0,
        twist_tip=0.0,
        airfoil="0012",
        n_span=12,
        n_chord=8,
    )


@pytest.fixture(scope="module")
def swept_mesh():
    """Swept tapered wing."""
    return make_wing_mesh(
        half_span=6.0,
        root_chord=3.0,
        tip_chord=1.0,
        sweep_le=35.0,
        dihedral=5.0,
        twist_tip=-3.0,
        airfoil="0012",
        n_span=10,
        n_chord=6,
    )


# ---------------------------------------------------------------------- #
# Unit-normal invariant                                                   #
# ---------------------------------------------------------------------- #

def test_unit_normals(rect_mesh):
    norms = np.linalg.norm(rect_mesh.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_unit_normals_swept(swept_mesh):
    norms = np.linalg.norm(swept_mesh.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


# ---------------------------------------------------------------------- #
# Outward-normal direction checks                                         #
# ---------------------------------------------------------------------- #

def test_upper_normals_point_up(rect_mesh):
    """For a wing with no dihedral, all upper-surface normals have nz > 0."""
    upper = rect_mesh.normals[rect_mesh.surface_id == SURF_UPPER]
    assert np.all(upper[:, 2] > 0), (
        f"Some upper normals do not point up; min nz = {upper[:, 2].min():.4f}"
    )


def test_lower_normals_point_down(rect_mesh):
    lower = rect_mesh.normals[rect_mesh.surface_id == SURF_LOWER]
    assert np.all(lower[:, 2] < 0), (
        f"Some lower normals do not point down; max nz = {lower[:, 2].max():.4f}"
    )


def test_tip_normals_point_outboard(rect_mesh):
    """Tip-cap normals of the right half-wing should have ny > 0."""
    tip = rect_mesh.normals[rect_mesh.surface_id == SURF_TIP]
    assert np.all(tip[:, 1] > 0), (
        f"Some tip normals do not point outboard; min ny = {tip[:, 1].min():.4f}"
    )


# ---------------------------------------------------------------------- #
# Panel areas                                                             #
# ---------------------------------------------------------------------- #

def test_areas_positive(rect_mesh):
    assert np.all(rect_mesh.areas > 0)


def test_areas_positive_swept(swept_mesh):
    assert np.all(swept_mesh.areas > 0)


# ---------------------------------------------------------------------- #
# TE-pair validity                                                         #
# ---------------------------------------------------------------------- #

def test_te_pairs_shape(rect_mesh):
    """TE pairs array has shape (n_span, 2)."""
    assert rect_mesh.te_pairs.ndim == 2
    assert rect_mesh.te_pairs.shape[1] == 2


def test_te_pairs_valid_indices(rect_mesh):
    Np = rect_mesh.n_panels
    assert rect_mesh.te_pairs.min() >= 0
    assert rect_mesh.te_pairs.max() < Np


def test_te_pairs_upper_is_upper(rect_mesh):
    """The first column of te_pairs must index upper-surface panels."""
    upper_ids = set(np.where(rect_mesh.surface_id == SURF_UPPER)[0])
    for idx in rect_mesh.te_pairs[:, 0]:
        assert int(idx) in upper_ids, f"TE upper panel {idx} is not an upper panel"


def test_te_pairs_lower_is_lower(rect_mesh):
    lower_ids = set(np.where(rect_mesh.surface_id == SURF_LOWER)[0])
    for idx in rect_mesh.te_pairs[:, 1]:
        assert int(idx) in lower_ids, f"TE lower panel {idx} is not a lower panel"


# ---------------------------------------------------------------------- #
# Panel count                                                             #
# ---------------------------------------------------------------------- #

def test_panel_count(rect_mesh):
    n_chord, n_span = 8, 12
    expected = n_chord * (2 * n_span + 1)
    assert rect_mesh.n_panels == expected


# ---------------------------------------------------------------------- #
# Wake seed                                                               #
# ---------------------------------------------------------------------- #

def test_wake_seed_shape(rect_mesh):
    assert rect_mesh.wake_seed.shape == (12, 3)  # n_span strips


def test_wake_seed_at_trailing_edge(rect_mesh):
    """Wake seeds should be at x ≈ root_chord (trailing-edge x position)."""
    root_chord = 2.0
    # For a rectangular wing with no sweep, x_TE = root_chord everywhere
    assert np.allclose(rect_mesh.wake_seed[:, 0], root_chord, atol=1e-6)


# ---------------------------------------------------------------------- #
# Near-planarity                                                          #
# ---------------------------------------------------------------------- #

def test_panels_nearly_planar_rect(rect_mesh):
    """Rectangular flat wing panels should be essentially planar."""
    assert rect_mesh.max_non_planarity() < 1e-10


def test_panels_nearly_planar_swept(swept_mesh):
    """Swept/dihedral wing may have some warp; keep below 1 % of chord."""
    tip_chord = 1.0
    assert swept_mesh.max_non_planarity() < 0.01 * tip_chord


# ---------------------------------------------------------------------- #
# Mesh constructor validation                                             #
# ---------------------------------------------------------------------- #

def test_bad_panel_index_raises():
    """Mesh constructor must reject out-of-range panel indices."""
    verts = np.zeros((4, 3))
    verts[0] = [0, 0, 0]
    verts[1] = [1, 0, 0]
    verts[2] = [1, 1, 0]
    verts[3] = [0, 1, 0]
    bad_panels = np.array([[0, 1, 2, 99]])   # index 99 out of range
    with pytest.raises(ValueError, match="out-of-range"):
        Mesh(
            vertices=verts,
            panels=bad_panels,
            te_pairs=np.empty((0, 2), dtype=int),
            wake_seed=np.empty((0, 3)),
        )
