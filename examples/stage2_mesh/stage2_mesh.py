"""Stage 2 demo — Mesh data structure and parametric wing generator.

Generates three wing configurations:
  1. Rectangular wing    (no sweep, no taper, no dihedral)
  2. Swept tapered wing  (35° LE sweep, taper ratio 0.4)
  3. Twisted flying wing (40° sweep, dihedral, tip twist, cambered airfoil)

Each mesh is visualised with outward normals and highlighted trailing-edge
panels, saved as a PNG, and exported as a VTK file for ParaView.

Run from the repo root:
    python examples/stage2_mesh/stage2_mesh.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves to file instead of screen
import matplotlib.pyplot as plt

from lspaero.geometry.wing import make_wing_mesh, SURF_UPPER, SURF_LOWER, SURF_TIP
from lspaero.viz import plot_mesh, save_vtk

OUT = Path(__file__).parent

# ---------------------------------------------------------------------- #
# Configurations                                                          #
# ---------------------------------------------------------------------- #

CONFIGS = [
    dict(
        label="Rectangular",
        kwargs=dict(
            half_span=5.0, root_chord=2.0, tip_chord=2.0,
            sweep_le=0.0, dihedral=0.0, twist_tip=0.0,
            airfoil="0012", n_span=16, n_chord=8,
        ),
    ),
    dict(
        label="Swept tapered",
        kwargs=dict(
            half_span=6.0, root_chord=3.0, tip_chord=1.2,
            sweep_le=35.0, dihedral=0.0, twist_tip=0.0,
            airfoil="0012", n_span=16, n_chord=8,
        ),
    ),
    dict(
        label="Flying wing",
        kwargs=dict(
            half_span=7.0, root_chord=3.5, tip_chord=1.0,
            sweep_le=42.0, dihedral=4.0, twist_tip=-4.0,
            airfoil="2412", n_span=20, n_chord=10,
        ),
    ),
]


def print_stats(label: str, mesh) -> None:
    n_up  = int((mesh.surface_id == SURF_UPPER).sum())
    n_lo  = int((mesh.surface_id == SURF_LOWER).sum())
    n_tip = int((mesh.surface_id == SURF_TIP).sum())
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  panels  : {mesh.n_panels}  (upper {n_up}, lower {n_lo}, tip {n_tip})")
    print(f"  vertices: {mesh.n_vertices}")
    print(f"  TE pairs: {len(mesh.te_pairs)}")
    print(f"  max non-planarity: {mesh.max_non_planarity():.2e} m")
    print(f"  is_watertight: {mesh.is_watertight()}")
    norms = np.linalg.norm(mesh.normals, axis=1)
    print(f"  normal magnitude: min {norms.min():.6f}  max {norms.max():.6f}")


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def main():
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw={"projection": "3d"},
    )

    for ax, cfg in zip(axes, CONFIGS):
        label = cfg["label"]
        mesh = make_wing_mesh(**cfg["kwargs"])
        print_stats(label, mesh)

        plot_mesh(
            mesh,
            ax=ax,
            show_normals=True,
            normal_stride=3,
            highlight_te=True,
            title=label,
            show=False,
        )

        vtk_path = OUT / f"stage2_{label.lower().replace(' ', '_')}.vtk"
        save_vtk(mesh, vtk_path)
        print(f"  VTK → {vtk_path}")

    fig.suptitle("Stage 2 — Mesh data structure & parametric wing generator",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out_png = OUT / "stage2_meshes.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_png}")


if __name__ == "__main__":
    main()
