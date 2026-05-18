"""Mesh visualisation: 3-D matplotlib plots and VTK legacy export.

All functions accept a :class:`~lspaero.geometry.Mesh` and produce either a
matplotlib figure or a ``.vtk`` file suitable for ParaView.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..geometry.mesh import Mesh
from ..geometry.wing import SURF_LOWER, SURF_TIP, SURF_UPPER

# RGBA colours per surface type
_SURF_RGBA = {
    SURF_UPPER: mc.to_rgba("#A8C7E8"),   # light blue
    SURF_LOWER: mc.to_rgba("#A8E8C7"),   # light green
    SURF_TIP:   mc.to_rgba("#E8D0A8"),   # light tan
}
_ALPHA = 0.75


def _make_poly(verts_list, face_rgba, edge_rgba, linewidth=0.3, alpha=_ALPHA):
    """Create a Poly3DCollection with explicit RGBA arrays (matplotlib 3.10 safe)."""
    n = len(verts_list)
    fa = np.array(face_rgba, dtype=float)
    if fa.ndim == 1:
        fa = np.tile(fa, (n, 1))
    fa[:, 3] = alpha                         # apply alpha to face colors
    ea = np.array(edge_rgba, dtype=float)
    if ea.ndim == 1:
        ea = np.tile(ea, (n, 1))
    poly = Poly3DCollection(verts_list)
    poly.set_facecolor(fa)
    poly.set_edgecolor(ea)
    poly.set_linewidth(linewidth)
    return poly


def plot_mesh(
    mesh: Mesh,
    ax: Optional[plt.Axes] = None,
    show_normals: bool = True,
    normal_scale: Optional[float] = None,
    normal_stride: int = 1,
    highlight_te: bool = True,
    title: str = "",
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """Render a Mesh in 3-D with normals and optional TE highlighting.

    Parameters
    ----------
    mesh : Mesh
    ax : Axes3D, optional
        Existing axes to draw into.  If *None*, a new figure is created.
    show_normals : bool
        Draw outward normal vectors as arrows.
    normal_scale : float, optional
        Arrow length.  Defaults to 3 % of the bounding-box diagonal.
    normal_stride : int
        Draw normals on every *stride*-th panel to reduce clutter.
    highlight_te : bool
        Outline trailing-edge panels in red.
    title : str
        Figure/axes title.
    show : bool
        Call ``plt.show()`` at the end.
    save_path : path-like, optional
        If given, save the figure at this path.

    Returns
    -------
    Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

    # Draw each surface group separately with a solid face colour (robust for
    # all matplotlib 3.x versions; avoids per-face colour list issues in 3D)
    has_sid = mesh.surface_id.size == mesh.n_panels
    group_ids = (
        np.unique(mesh.surface_id) if has_sid
        else np.array([-1])
    )
    default_rgba = mc.to_rgba("#A8C7E8")

    for sid in group_ids:
        if has_sid:
            mask = mesh.surface_id == sid
            rgba = _SURF_RGBA.get(int(sid), default_rgba)
        else:
            mask = np.ones(mesh.n_panels, dtype=bool)
            rgba = default_rgba

        verts_list = list(mesh.vertices[mesh.panels[mask]])
        if not verts_list:
            continue
        poly = _make_poly(verts_list, rgba, (0.1, 0.1, 0.1, 1.0), linewidth=0.3)
        ax.add_collection3d(poly)

    # TE highlighting — draw TE panel edges in red
    if highlight_te and mesh.te_pairs.size > 0:
        te_idx = mesh.te_pairs.ravel()
        te_verts_list = list(mesh.vertices[mesh.panels[te_idx]])
        red_rgba = np.array([1.0, 0.0, 0.0, 1.0])
        te_poly = _make_poly(
            te_verts_list, (0, 0, 0, 0), red_rgba, linewidth=1.5, alpha=0.0
        )
        ax.add_collection3d(te_poly)

    # Normal arrows (subsampled)
    if show_normals:
        if normal_scale is None:
            bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
            normal_scale = float(np.linalg.norm(bbox)) * 0.03

        idx = np.arange(0, mesh.n_panels, normal_stride)
        c = mesh.centroids[idx]
        n = mesh.normals[idx]
        ax.quiver(
            c[:, 0], c[:, 1], c[:, 2],
            n[:, 0], n[:, 1], n[:, 2],
            length=normal_scale,
            color="navy",
            normalize=True,
            linewidth=0.6,
            arrow_length_ratio=0.3,
        )

    # Equal-aspect bounding box
    lo = mesh.vertices.min(axis=0)
    hi = mesh.vertices.max(axis=0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max() * 1.1
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_xlabel("x (chordwise)")
    ax.set_ylabel("y (spanwise)")
    ax.set_zlabel("z (up)")
    if title:
        ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def save_vtk(mesh: Mesh, filepath: str | Path) -> None:
    """Export a Mesh to VTK legacy ASCII format (ParaView / VisIt compatible).

    The file contains:
    - POINTS: vertex coordinates
    - CELLS: quad connectivity
    - CELL_TYPES: VTK_QUAD (9)
    - CELL_DATA / SCALARS: surface_id
    - CELL_DATA / VECTORS: outward normals

    Parameters
    ----------
    mesh : Mesh
    filepath : path-like
        Output file path.  The ``.vtk`` extension is conventional.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    Nv = mesh.n_vertices
    Np = mesh.n_panels

    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "LSPAero Wing Mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "",
        f"POINTS {Nv} float",
    ]
    for v in mesh.vertices:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    lines += ["", f"CELLS {Np} {Np * 5}"]
    for p in mesh.panels:
        lines.append(f"4 {p[0]} {p[1]} {p[2]} {p[3]}")

    lines += ["", f"CELL_TYPES {Np}"]
    lines += ["9"] * Np   # VTK_QUAD = 9

    lines += [
        "",
        f"CELL_DATA {Np}",
        "SCALARS surface_id int 1",
        "LOOKUP_TABLE default",
    ]
    if mesh.surface_id.size == Np:
        for sid in mesh.surface_id:
            lines.append(str(int(sid)))
    else:
        lines += ["0"] * Np

    lines += ["", "VECTORS normals float"]
    for n in mesh.normals:
        lines.append(f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    filepath.write_text("\n".join(lines) + "\n")
