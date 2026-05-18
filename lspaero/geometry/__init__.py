"""Geometry module: airfoil/wing generators and Mesh data structure."""

from .mesh import Mesh
from .naca import naca4, naca4_surfaces
from .wing import SURF_LOWER, SURF_TIP, SURF_UPPER, make_wing_mesh

__all__ = [
    "Mesh",
    "naca4",
    "naca4_surfaces",
    "make_wing_mesh",
    "SURF_UPPER",
    "SURF_LOWER",
    "SURF_TIP",
]
