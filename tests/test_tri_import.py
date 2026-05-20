"""Tests for Cart3D .tri import pipeline (Stage 7).

Covers:
- read_tri round-trip
- Degenerate-quad area and normal (triangle stored as (i,j,k,k))
- detect_te_edges on the real OpenVSP wing.tri
- tris_to_mesh produces a valid Mesh for each case
"""
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lspaero.io.tri import read_tri, read_tkey
from lspaero.geometry.tri_utils import (
    build_edge_table,
    detect_te_edges,
    tris_to_mesh,
)
from lspaero.geometry.mesh import Mesh

# Paths to the OpenVSP sample meshes
MESH_DIR = Path(__file__).parent.parent / "examples" / "openvsp_mesh"
WING_TRI  = MESH_DIR / "wing.tri"
WBODY_TRI = MESH_DIR / "wing-body.tri"


# ---------------------------------------------------------------------------
# read_tri
# ---------------------------------------------------------------------------

def _write_minimal_tri(tmp_path: Path) -> Path:
    """Write a minimal 2-triangle .tri file (unit square split into 2 tris)."""
    content = """\
4 2
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
1 2 3
1 3 4
1
2
"""
    p = tmp_path / "test.tri"
    p.write_text(content)
    return p


def test_read_tri_roundtrip(tmp_path):
    p = _write_minimal_tri(tmp_path)
    verts, tris, comp_ids = read_tri(p)

    assert verts.shape == (4, 3)
    assert tris.shape  == (2, 3)
    assert comp_ids.shape == (2,)

    # 0-based check
    assert tris.min() == 0
    assert tris.max() == 3

    # comp IDs
    assert list(comp_ids) == [1, 2]


def test_read_tkey(tmp_path):
    p = tmp_path / "test.tkey"
    p.write_text("# VSP Tag Key File\n./test.tri\n2\n\n1,Surf0\n2,Surf1\n")
    names = read_tkey(p)
    assert names == {1: "Surf0", 2: "Surf1"}


# ---------------------------------------------------------------------------
# Degenerate quad: area, normal, centroid
# ---------------------------------------------------------------------------

def test_degenerate_quad_area_normal():
    """Triangle (0,0,0)–(1,0,0)–(0,1,0) stored as degenerate quad."""
    verts = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
    ])
    # Degenerate quad: repeat last vertex
    panels = np.array([[0, 1, 2, 2]])

    mesh = Mesh(
        vertices=verts,
        panels=panels,
        te_pairs=np.empty((0, 2), dtype=int),
        wake_seed=np.empty((0, 3)),
    )
    assert np.isclose(mesh.areas[0], 0.5), f"area={mesh.areas[0]}"
    assert np.allclose(mesh.normals[0], [0., 0., 1.]), f"normal={mesh.normals[0]}"


def test_degenerate_quad_centroid():
    """Centroid of triangle (0,0,0)–(3,0,0)–(0,3,0) must be (1,1,0)."""
    verts = np.array([
        [0., 0., 0.],
        [3., 0., 0.],
        [0., 3., 0.],
    ])
    panels = np.array([[0, 1, 2, 2]])
    mesh = Mesh(
        vertices=verts,
        panels=panels,
        te_pairs=np.empty((0, 2), dtype=int),
        wake_seed=np.empty((0, 3)),
    )
    assert np.allclose(mesh.centroids[0], [1., 1., 0.]), f"centroid={mesh.centroids[0]}"


def test_unit_square_two_tris():
    """Unit square split into 2 triangles: total area = 1, normals both [0,0,1]."""
    verts = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
    ])
    panels = np.array([
        [0, 1, 2, 2],   # lower-right triangle
        [0, 2, 3, 3],   # upper-left triangle
    ])
    mesh = Mesh(
        vertices=verts,
        panels=panels,
        te_pairs=np.empty((0, 2), dtype=int),
        wake_seed=np.empty((0, 3)),
    )
    assert np.isclose(mesh.areas.sum(), 1.0)
    assert np.allclose(mesh.normals, [[0., 0., 1.], [0., 0., 1.]])


# ---------------------------------------------------------------------------
# detect_te_edges on real OpenVSP meshes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WING_TRI.exists(), reason="wing.tri not found")
def test_detect_te_wing():
    verts, tris, comp_ids = read_tri(WING_TRI)

    # Use only right wing half (comp 1, y >= 0)
    mask = comp_ids == 1
    te_pairs, te_verts, wake_seed = detect_te_edges(
        verts, tris[mask], comp_ids[mask], lifting_comp_ids=[1]
    )

    # Wing has 5 spanwise strips → 5 TE edges for the right half
    assert len(te_pairs) == 5, f"expected 5 TE pairs, got {len(te_pairs)}"

    # Upper panel z-centroid must be higher than lower panel
    all_tris = tris[mask]
    for up, lo in te_pairs:
        z_up = verts[all_tris[up]].mean(axis=0)[2]
        z_lo = verts[all_tris[lo]].mean(axis=0)[2]
        assert z_up > z_lo, "upper panel should have higher z centroid"

    # TE verts: inner |y| < outer |y|
    for k in range(len(te_pairs)):
        assert abs(te_verts[k, 0, 1]) <= abs(te_verts[k, 1, 1])

    # wake_seed = midpoint of the two TE endpoints
    expected_seed = 0.5 * (te_verts[:, 0] + te_verts[:, 1])
    assert np.allclose(wake_seed, expected_seed)


@pytest.mark.skipif(not WBODY_TRI.exists(), reason="wing-body.tri not found")
def test_detect_te_wing_body():
    verts, tris, comp_ids = read_tri(WBODY_TRI)

    # Right wing half only (comp 1)
    mask = comp_ids == 1
    te_pairs, te_verts, wake_seed = detect_te_edges(
        verts, tris[mask], comp_ids[mask], lifting_comp_ids=[1]
    )

    # Wing-body wing has 5 spanwise strips for the right half
    assert len(te_pairs) == 5, f"expected 5 TE pairs, got {len(te_pairs)}"


# ---------------------------------------------------------------------------
# tris_to_mesh produces a valid Mesh
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WING_TRI.exists(), reason="wing.tri not found")
def test_tris_to_mesh_wing():
    verts, tris, comp_ids = read_tri(WING_TRI)

    mask = comp_ids == 1
    n_raw = int(mask.sum())

    # Default: tip-cap panels are excluded → panel count < raw triangle count
    mesh = tris_to_mesh(
        verts, tris, comp_ids,
        lifting_comp_ids=[1],
        mask=mask,
    )

    assert isinstance(mesh, Mesh)
    assert mesh.n_panels < n_raw, (
        f"tip-cap removal should reduce panel count "
        f"({mesh.n_panels} vs {n_raw} raw)"
    )
    assert mesh.panels.shape[1] == 4
    assert mesh.te_pairs.shape == (5, 2)
    assert mesh.te_verts is not None
    assert mesh.te_verts.shape == (5, 2, 3)
    assert np.all(mesh.areas > 0)

    # With tip cap kept: panel count matches raw triangle count
    mesh_full = tris_to_mesh(
        verts, tris, comp_ids,
        lifting_comp_ids=[1],
        mask=mask,
        exclude_tip_cap=False,
    )
    assert mesh_full.n_panels == n_raw


@pytest.mark.skipif(not WBODY_TRI.exists(), reason="wing-body.tri not found")
def test_tris_to_mesh_wing_body():
    verts, tris, comp_ids = read_tri(WBODY_TRI)

    # Right wing only
    mask = comp_ids == 1
    mesh = tris_to_mesh(
        verts, tris, comp_ids,
        lifting_comp_ids=[1],
        mask=mask,
    )

    assert isinstance(mesh, Mesh)
    assert mesh.te_pairs.shape[0] == 5
    assert mesh.te_verts is not None
