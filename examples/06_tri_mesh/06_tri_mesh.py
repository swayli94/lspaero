"""Stage 7: Triangulated mesh support — quad vs tri validation.

Validates that the degenerate-quad representation in ``triangulate_mesh``
and the parallelogram-completion branch in ``_ring_points`` give aerodynamic
results that match the parent quad mesh.

Three test cases
----------------
1. **Rectangular VLM** (AR = 8, flat plate)
   ``solve_vlm`` — expected CL error: **0.000%** (exact, rectangle = parallelogram).

2. **Thick rectangular wing** (NACA 0012, AR = 8)
   ``solve_morino`` — expected CL error: **0.000%** (same reason).

3. **Swept/tapered VLM** (Λ = 30°, taper = 0.5, AR ≈ 6.7)
   ``solve_vlm`` — expected CL error: **~5–6%** (inherent parallelogram-completion
   approximation; bounded, documented, not a defect).

S_ref note
----------
The degenerate-quad area equals the area of the first-diagonal triangle
(v0, v1, v2), which is NOT in general half the parent-quad area for
trapezoidal (tapered) panels.  **Always pass an explicit S_ref** derived
from the original quad mesh to avoid doubled/halved CL artefacts.

Outputs
-------
06_mesh_comparison.png   — panel centroids and normals for quad vs tri
06_CL_vs_alpha.png       — CL(α) overlay for all three cases with error table
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lspaero.geometry.wing import make_vlm_mesh, make_wing_mesh
from lspaero.geometry.tri_utils import triangulate_mesh
from lspaero.solver.solve import solve_vlm
from lspaero.solver.morino import solve_morino

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Geometry parameters
# ---------------------------------------------------------------------------
ALPHAS = np.array([-4.0, 0.0, 4.0, 8.0, 12.0])

# Case 1 & 2: rectangular AR = 8
RECT_HALF_SPAN   = 4.0
RECT_CHORD       = 1.0
RECT_S_REF_PHYS  = RECT_HALF_SPAN * RECT_CHORD   # = 4.0  (half-wing area)

# Case 3: swept/tapered
SWT_HALF_SPAN    = 5.0
SWT_ROOT_CHORD   = 2.0
SWT_TIP_CHORD    = 1.0
SWT_SWEEP        = 30.0   # degrees, LE
SWT_S_REF_PHYS   = 0.5 * (SWT_ROOT_CHORD + SWT_TIP_CHORD) * SWT_HALF_SPAN  # = 7.5 (trapezoidal)

# ---------------------------------------------------------------------------
# Build meshes
# ---------------------------------------------------------------------------
print("── Building meshes …")

# Case 1: rectangular VLM
mesh_rv_q = make_vlm_mesh(half_span=RECT_HALF_SPAN, root_chord=RECT_CHORD,
                           tip_chord=RECT_CHORD, n_span=8, n_chord=1)
mesh_rv_t = triangulate_mesh(mesh_rv_q)
print(f"   Rect VLM:  quad {mesh_rv_q.n_panels} panels  "
      f"→ tri {mesh_rv_t.n_panels} panels  "
      f"(S_ref_tri={mesh_rv_t.areas.sum():.3f}, "
      f"S_ref_quad={mesh_rv_q.areas.sum():.3f})")

# Case 2: rectangular thick
mesh_rt_q = make_wing_mesh(half_span=RECT_HALF_SPAN, root_chord=RECT_CHORD,
                            tip_chord=RECT_CHORD, airfoil="0012",
                            n_span=6, n_chord=4, include_tip_cap=False)
mesh_rt_t = triangulate_mesh(mesh_rt_q)
S_ref_rt = float(mesh_rt_q.areas[mesh_rt_q.surface_id == 0].sum())
print(f"   Rect thick: quad {mesh_rt_q.n_panels} panels  "
      f"→ tri {mesh_rt_t.n_panels} panels  "
      f"(upper S_ref={S_ref_rt:.3f})")

# Case 3: swept VLM
mesh_sw_q = make_vlm_mesh(half_span=SWT_HALF_SPAN, root_chord=SWT_ROOT_CHORD,
                           tip_chord=SWT_TIP_CHORD, sweep_le=SWT_SWEEP,
                           n_span=6, n_chord=4)
mesh_sw_t = triangulate_mesh(mesh_sw_q)
S_ref_sw = float(mesh_sw_q.areas.sum())
print(f"   Swept VLM: quad {mesh_sw_q.n_panels} panels  "
      f"→ tri {mesh_sw_t.n_panels} panels  "
      f"(S_ref_quad={S_ref_sw:.3f}, S_ref_tri={mesh_sw_t.areas.sum():.3f})")

# ---------------------------------------------------------------------------
# CL(α) sweeps
# ---------------------------------------------------------------------------
def cl_sweep(mesh, solver_fn, s_ref, label):
    """Run α sweep, return CL array and print results."""
    CLs = []
    for a in ALPHAS:
        r = solver_fn(mesh, alpha_deg=float(a), S_ref=s_ref)
        CLs.append(r["CL"])
    return np.array(CLs)


print("\n── Case 1: rectangular VLM (AR=8, flat plate)")
CL_rv_q = cl_sweep(mesh_rv_q, solve_vlm, RECT_S_REF_PHYS, "quad")
CL_rv_t = cl_sweep(mesh_rv_t, solve_vlm, RECT_S_REF_PHYS, "tri ")
slope_rv_q = np.polyfit(np.radians(ALPHAS), CL_rv_q, 1)[0]
slope_rv_t = np.polyfit(np.radians(ALPHAS), CL_rv_t, 1)[0]
err_rv = 100 * abs(slope_rv_t - slope_rv_q) / abs(slope_rv_q)
print(f"   CL_α: quad={slope_rv_q:.4f}/rad   tri={slope_rv_t:.4f}/rad   "
      f"slope error={err_rv:.4f}%")

print("\n── Case 2: rectangular thick, NACA 0012, solve_morino")
CL_rt_q = cl_sweep(mesh_rt_q, solve_morino, S_ref_rt, "quad")
CL_rt_t = cl_sweep(mesh_rt_t, solve_morino, S_ref_rt, "tri ")
slope_rt_q = np.polyfit(np.radians(ALPHAS), CL_rt_q, 1)[0]
slope_rt_t = np.polyfit(np.radians(ALPHAS), CL_rt_t, 1)[0]
err_rt = 100 * abs(slope_rt_t - slope_rt_q) / abs(slope_rt_q)
print(f"   CL_α: quad={slope_rt_q:.4f}/rad   tri={slope_rt_t:.4f}/rad   "
      f"slope error={err_rt:.4f}%")

print("\n── Case 3: swept/tapered VLM (Λ=30°, taper=0.5)")
CL_sw_q = cl_sweep(mesh_sw_q, solve_vlm, S_ref_sw, "quad")
CL_sw_t = cl_sweep(mesh_sw_t, solve_vlm, S_ref_sw, "tri ")
slope_sw_q = np.polyfit(np.radians(ALPHAS), CL_sw_q, 1)[0]
slope_sw_t = np.polyfit(np.radians(ALPHAS), CL_sw_t, 1)[0]
err_sw = 100 * abs(slope_sw_t - slope_sw_q) / abs(slope_sw_q)
print(f"   CL_α: quad={slope_sw_q:.4f}/rad   tri={slope_sw_t:.4f}/rad   "
      f"slope error={err_sw:.4f}%  (parallelogram-completion approx.)")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# --- Figure 1: mesh comparison (centroids + normals) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
cases = [
    (mesh_rv_q, mesh_rv_t, "Rect VLM (AR=8)"),
    (mesh_rt_q, mesh_rt_t, "Rect thick NACA 0012"),
    (mesh_sw_q, mesh_sw_t, "Swept VLM (Λ=30°, taper 0.5)"),
]
for ax, (mq, mt, title) in zip(axes, cases):
    ax.scatter(mq.centroids[:, 1], mq.centroids[:, 0],
               s=20, c="steelblue", alpha=0.7, label="quad")
    ax.scatter(mt.centroids[:, 1], mt.centroids[:, 0],
               s=8,  c="tomato",    alpha=0.7, label="tri", marker="x")

    # normals on quad (subsampled)
    step = max(1, mq.n_panels // 80)
    idx  = np.arange(0, mq.n_panels, step)
    scale = 0.15
    ax.quiver(mq.centroids[idx, 1], mq.centroids[idx, 0],
              mq.normals[idx, 1] * scale, mq.normals[idx, 2] * scale,
              angles="xy", scale_units="xy", scale=1,
              color="steelblue", width=0.003, alpha=0.6)

    ax.set_xlabel("y (span)"); ax.set_ylabel("x (chord)")
    ax.set_title(f"{title}\n{mq.n_panels} quad  /  {mt.n_panels} tri panels")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.legend(fontsize=7, markerscale=1.5)

plt.tight_layout()
fig.savefig(HERE / "06_mesh_comparison.png", dpi=150)
plt.close(fig)
print("\n→ saved 06_mesh_comparison.png")

# --- Figure 2: CL(α) overlay, all three cases ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

case_data = [
    ("Rect VLM (AR=8)",          CL_rv_q, CL_rv_t, err_rv,
     "solve_vlm",    "0.000% expected"),
    ("Rect thick NACA 0012",      CL_rt_q, CL_rt_t, err_rt,
     "solve_morino", "0.000% expected"),
    ("Swept VLM (Λ=30°, t=0.5)", CL_sw_q, CL_sw_t, err_sw,
     "solve_vlm",    "~5% (parallelogram approx.)"),
]

for ax, (title, CL_q, CL_t, err, solver, note) in zip(axes, case_data):
    ax.plot(ALPHAS, CL_q, "o-",  label=f"quad  CL_α={np.polyfit(np.radians(ALPHAS), CL_q, 1)[0]:.3f}/rad")
    ax.plot(ALPHAS, CL_t, "s--", label=f"tri   CL_α={np.polyfit(np.radians(ALPHAS), CL_t, 1)[0]:.3f}/rad")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("α (deg)")
    ax.set_ylabel("CL")
    ax.set_title(f"{title}\n{solver}  |slope error| = {err:.3f}%\n({note})")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)

plt.tight_layout()
fig.savefig(HERE / "06_CL_vs_alpha.png", dpi=150)
plt.close(fig)
print("→ saved 06_CL_vs_alpha.png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"{'Case':<30} {'Solver':<13} {'quad CL_α':>10} {'tri CL_α':>10} {'err%':>7}")
print("-" * 65)
for title, CL_q, CL_t, err, solver, _ in case_data:
    s_q = np.polyfit(np.radians(ALPHAS), CL_q, 1)[0]
    s_t = np.polyfit(np.radians(ALPHAS), CL_t, 1)[0]
    print(f"{title:<30} {solver:<13} {s_q:>10.4f} {s_t:>10.4f} {err:>7.3f}%")
print("=" * 65)
print("\n✓ Stage 7 example complete.")
