"""Geometry module: airfoil/wing generators, fuselage generator, and Mesh data structure."""

from .body import SURF_BODY, make_body_mesh, sears_haack_profile
from .mesh import Mesh, combine_meshes
from .naca import naca4, naca4_surfaces
from .wing import SURF_LOWER, SURF_TIP, SURF_UPPER, make_vlm_mesh, make_wing_mesh

__all__ = [
    # Mesh data structure
    "Mesh",
    "combine_meshes",
    # Wing generators
    "make_wing_mesh",
    "make_vlm_mesh",
    "SURF_UPPER",
    "SURF_LOWER",
    "SURF_TIP",
    # Body generator
    "make_body_mesh",
    "sears_haack_profile",
    "SURF_BODY",
    # NACA airfoil
    "naca4",
    "naca4_surfaces",
]
