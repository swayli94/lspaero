"""Cart3D ASCII ``.tri`` file reader.

OpenVSP exports surface meshes in the Cart3D ``.tri`` format.  The file
layout is:

::

    Line 1:               Nvert  Ntri
    Lines 2 … Nvert+1:   x  y  z          (one vertex per line)
    Lines Nvert+2 … Nvert+Ntri+1:  i  j  k (1-based triangle connectivity)
    Lines Nvert+Ntri+2 … end:  comp_id     (one integer per line)

A companion ``.tkey`` file maps integer component IDs to surface names::

    # VSP Tag Key File
    ./mesh.tri
    <Ncomp>

    1,WingGeom_S_Surf0
    2,WingGeom_S_Surf1
    ...

Reference: Cart3D User Manual, NASA Ames Research Center.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_tri(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a Cart3D ASCII ``.tri`` file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.tri`` file.

    Returns
    -------
    vertices : (Nv, 3) float
        Vertex coordinates ``[x, y, z]``.
    triangles : (Nt, 3) int
        Triangle vertex indices, **0-based**.
    comp_ids : (Nt,) int
        Integer component ID for each triangle (1-based as in the file).
    """
    path = Path(path)
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    # --- header -----------------------------------------------------------
    nv, nt = int(lines[0].split()[0]), int(lines[0].split()[1])

    # --- vertices ---------------------------------------------------------
    vertices = np.empty((nv, 3), dtype=float)
    for i in range(nv):
        vertices[i] = list(map(float, lines[1 + i].split()))

    # --- triangles (convert from 1-based to 0-based) ----------------------
    triangles = np.empty((nt, 3), dtype=int)
    for i in range(nt):
        triangles[i] = list(map(int, lines[1 + nv + i].split()))
    triangles -= 1

    # --- component IDs ----------------------------------------------------
    comp_ids = np.array(
        [int(lines[1 + nv + nt + i]) for i in range(nt)],
        dtype=int,
    )

    return vertices, triangles, comp_ids


def read_tkey(path: str | Path) -> dict[int, str]:
    """Read a Cart3D ``.tkey`` tag-key file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.tkey`` file (same stem as the ``.tri`` file).

    Returns
    -------
    comp_names : dict[int → str]
        Mapping from component ID to surface name string.
    """
    path = Path(path)
    comp_names: dict[int, str] = {}
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("."):
                continue
            if ln.isdigit():          # lone line with Ncomp count
                continue
            if "," in ln:
                cid_str, name = ln.split(",", 1)
                try:
                    comp_names[int(cid_str)] = name.strip()
                except ValueError:
                    pass
    return comp_names
