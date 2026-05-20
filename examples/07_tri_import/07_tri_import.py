"""Stage 8: Cart3D .tri import — OpenVSP wing aerodynamic comparison.

This script demonstrates the full import pipeline:
  1. Read OpenVSP .tri meshes (wing-only and wing-body).
  2. Detect trailing edges automatically (crease-angle method).
  3. Assemble lspaero Mesh objects (degenerate-quad representation).
  4. Run ``solve_morino`` on the imported mesh using an external VLM camber
     mesh that matches the wing planform.
  5. Compare the imported-mesh result against a parametric lspaero wing
     with identical planform and airfoil.

**Why an external cam_mesh?**
  OpenVSP exports triangular panels whose vertex ordering (v0, v1, v2) does
  not follow the parametric convention assumed by ``_ring_points`` (which
  expects v0 = fwd-inner, v1 = aft-inner, v2 = aft-outer).  Using the
  imported panels directly for VLM would produce incorrect bound-vortex
  endpoints A and B (the parallelogram completion would be wrong).
  Instead, we supply a ``make_vlm_mesh`` camber mesh with the same planform
  — its structured vertex ordering is correct — and use the imported thick
  surface only for the Hess-Smith source solve (thickness Cp).

Stage 7 (tri mesh validation) established that triangulated meshes give
**exact** CL agreement with quad meshes for rectangular wings and **~5–6%**
error for strongly tapered wings (inherent parallelogram-completion
approximation).  This stage exercises the full I/O pipeline on real
OpenVSP meshes; the 5% tolerance applies.

Wing geometry (from OpenVSP model):
  half-span  = 9.0
  root chord = 4.0    tip chord = 1.0    taper = 0.25
  LE sweep   = 30.0°                     airfoil = NACA 0010
  Reference area S_ref = 0.5 * (4+1) * 9 = 22.5  (trapezoidal half-wing)

Outputs
-------
  07_mesh_comparison.png  — imported tri mesh vs parametric mesh (normals)
  07_CL_vs_alpha.png      — CL(α) and CDi(α) comparison with % error annotation
  07_Cp_wing.png          — chordwise Cp at mid-span, both meshes, α = 5°
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

from lspaero.io.tri import read_tri, read_tkey
from lspaero.geometry.tri_utils import tris_to_mesh
from lspaero.geometry.wing import make_wing_mesh, make_vlm_mesh
from lspaero.solver.morino import solve_morino

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = Path(__file__).parent
MESH_DIR = HERE.parent / "openvsp_mesh"
WING_TRI  = MESH_DIR / "wing.tri"
WBODY_TRI = MESH_DIR / "wing-body.tri"

# Wing parameters (measured from OpenVSP mesh)
HALF_SPAN   = 9.0
ROOT_CHORD  = 4.0
TIP_CHORD   = 1.0
SWEEP_LE    = 30.0          # degrees, LE
AIRFOIL     = "0010"        # NACA 0010 (t/c ≈ 10%)
S_REF       = 0.5 * (ROOT_CHORD + TIP_CHORD) * HALF_SPAN   # = 22.5

# ---------------------------------------------------------------------------
# 1.  Import OpenVSP wing (right half, comp 1)
# ---------------------------------------------------------------------------
print("── Reading wing.tri …")
verts, tris, comp_ids = read_tri(WING_TRI)

comp_names = read_tkey(MESH_DIR / "wing.tkey")
print(f"   Components: {comp_names}")
print(f"   Vertices: {len(verts)}   Triangles: {len(tris)}")

# Restrict to right wing half (comp 1)
mask_wing = comp_ids == 1
mesh_tri = tris_to_mesh(
    verts, tris, comp_ids,
    lifting_comp_ids=[1],
    mask=mask_wing,
)
print(f"   Tri mesh: {mesh_tri.n_panels} panels, "
      f"{len(mesh_tri.te_pairs)} TE pairs")

# ---------------------------------------------------------------------------
# 2.  Parametric wing with matching planform and VLM camber mesh
# ---------------------------------------------------------------------------
print("\n── Building parametric wing and VLM camber mesh …")
mesh_par = make_wing_mesh(
    half_span=HALF_SPAN,
    root_chord=ROOT_CHORD,
    tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE,
    airfoil=AIRFOIL,
    n_span=5,       # match OpenVSP's 5 spanwise strips
    n_chord=10,
    include_tip_cap=False,
)
print(f"   Parametric mesh: {mesh_par.n_panels} panels, "
      f"{len(mesh_par.te_pairs)} TE pairs")

# External VLM camber mesh — used for solve_morino on the imported mesh.
# make_vlm_mesh follows the parametric vertex convention (v0=fwd-inner,
# v1=aft-inner) so _ring_points gives correct A, B for VLM.
cam_mesh = make_vlm_mesh(
    half_span=HALF_SPAN,
    root_chord=ROOT_CHORD,
    tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE,
    n_span=5,
    n_chord=10,
)
print(f"   VLM cam mesh:   {cam_mesh.n_panels} panels")

# ---------------------------------------------------------------------------
# 3.  CL / CDi vs α sweep
# ---------------------------------------------------------------------------
alphas = np.linspace(-4, 12, 9)


def run_sweep(mesh, label, s_ref=S_REF, ext_cam=None):
    """Run α sweep with solve_morino; return (CLs, CDis)."""
    CLs, CDis = [], []
    kw = {} if ext_cam is None else {"cam_mesh": ext_cam}
    for a in alphas:
        res = solve_morino(mesh, alpha_deg=float(a),
                           S_ref=s_ref, b_ref=2 * HALF_SPAN, **kw)
        CLs.append(res["CL"])
        CDis.append(res["CDi"])
        print(f"   {label}  α={a:+5.1f}°  CL={res['CL']:.4f}  CDi={res['CDi']:.5f}")
    return np.array(CLs), np.array(CDis)


print("\n── Sweep: imported tri mesh (solve_morino, external cam_mesh)")
CL_tri, CDi_tri = run_sweep(mesh_tri, "tri ", ext_cam=cam_mesh)

print("\n── Sweep: parametric mesh (solve_morino, internal camber mesh)")
CL_par, CDi_par = run_sweep(mesh_par, "par ")

# CL slope comparison
alpha_rad = np.radians(alphas)
slope_tri = np.polyfit(alpha_rad, CL_tri, 1)[0]
slope_par = np.polyfit(alpha_rad, CL_par, 1)[0]
err_pct   = 100.0 * abs(slope_tri - slope_par) / slope_par

print(f"\n── CL_α:  tri = {slope_tri:.3f}/rad   "
      f"par = {slope_par:.3f}/rad   "
      f"error = {err_pct:.1f} %  (target < 5%)")

# ---------------------------------------------------------------------------
# 4.  Cp chordwise at mid-span, α = 5°
# ---------------------------------------------------------------------------
alpha_cp = 5.0
res_tri  = solve_morino(mesh_tri, alpha_deg=alpha_cp,
                        S_ref=S_REF, b_ref=2 * HALF_SPAN, cam_mesh=cam_mesh)
res_par  = solve_morino(mesh_par, alpha_deg=alpha_cp,
                        S_ref=S_REF, b_ref=2 * HALF_SPAN)

# ---------------------------------------------------------------------------
# 5.  Figures
# ---------------------------------------------------------------------------

# --- Figure 1: mesh comparison (panel centroids coloured by surface_id / z) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, mesh, title in zip(axes,
                            [mesh_tri, mesh_par],
                            ["Imported tri mesh (comp 1)", "Parametric mesh"]):
    # colour by z-centroid for imported (no meaningful surface_id 0/1)
    c_vals = mesh.centroids[:, 2]
    sc = ax.scatter(mesh.centroids[:, 1], mesh.centroids[:, 0],
                    c=c_vals, cmap="RdBu_r", s=6, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="z (m)")
    # normals (subsampled)
    step = max(1, mesh.n_panels // 200)
    idx  = np.arange(0, mesh.n_panels, step)
    scale = 0.3
    ax.quiver(mesh.centroids[idx, 1], mesh.centroids[idx, 0],
              mesh.normals[idx, 1] * scale, mesh.normals[idx, 2] * scale,
              angles="xy", scale_units="xy", scale=1, color="k", width=0.002,
              alpha=0.5)
    ax.set_xlabel("y (span)"); ax.set_ylabel("x (chord)")
    ax.set_title(f"{title}\n{mesh.n_panels} panels, "
                 f"{len(mesh.te_pairs)} TE strips")
    ax.set_aspect("equal")
    ax.invert_yaxis()
plt.tight_layout()
fig.savefig(HERE / "07_mesh_comparison.png", dpi=150)
plt.close(fig)
print("\n→ saved 07_mesh_comparison.png")

# --- Figure 2: CL and CDi vs alpha ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(alphas, CL_tri,  "o-",  label=f"Imported tri  CL_α={slope_tri:.3f}/rad")
ax.plot(alphas, CL_par,  "s--", label=f"Parametric    CL_α={slope_par:.3f}/rad")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("α (deg)")
ax.set_ylabel("CL")
ax.set_title(f"CL vs α — error = {err_pct:.1f} %  (target < 5%)")
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.5)

ax = axes[1]
ax.plot(alphas, CDi_tri, "o-",  label="Imported tri")
ax.plot(alphas, CDi_par, "s--", label="Parametric")
ax.set_xlabel("α (deg)")
ax.set_ylabel("CDi")
ax.set_title("CDi vs α")
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.5)

plt.tight_layout()
fig.savefig(HERE / "07_CL_vs_alpha.png", dpi=150)
plt.close(fig)
print("→ saved 07_CL_vs_alpha.png")

# --- Figure 3: Cp comparison at mid-span strip (α = 5°) ---
# Both meshes are interpolated to the SAME y value by blending the two
# nearest spanwise strips.  This avoids any strip-alignment mismatch.

def _cp_at_y(cy: np.ndarray, cx: np.ndarray, cz: np.ndarray,
             Cp: np.ndarray, y_target: float,
             n_xc: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Linearly interpolate upper/lower Cp to an exact spanwise position.

    Finds the two strips bracketing *y_target* (or the single strip if an
    exact match exists), blends their Cp values with weights proportional to
    the fractional distance, and returns the result on a uniform x/c grid.

    Parameters
    ----------
    cy, cx, cz : (N,) arrays — panel centroid coordinates.
    Cp         : (N,) array  — panel pressure coefficients.
    y_target   : target spanwise station.
    n_xc       : number of points in the output x/c grid.

    Returns
    -------
    (xc, cp_upper, cp_lower) each of length *n_xc*, or None on failure.
    """
    y_rounded = np.round(cy, 1)
    y_uniq    = np.unique(y_rounded)

    # Strips that bracket y_target
    below = y_uniq[y_uniq <= y_target + 1e-6]
    above = y_uniq[y_uniq >= y_target - 1e-6]
    if len(below) == 0 or len(above) == 0:
        return None

    y1, y2 = float(below[-1]), float(above[0])
    if abs(y2 - y1) < 1e-6:
        weights = [(y1, 1.0)]               # exact match — no blending needed
    else:
        w2 = (y_target - y1) / (y2 - y1)   # linear interpolation weight
        weights = [(y1, 1.0 - w2), (y2, w2)]

    xc_common = np.linspace(0.0, 1.0, n_xc)
    cp_up = np.zeros(n_xc)
    cp_lo = np.zeros(n_xc)
    w_sum = 0.0

    for y_s, w in weights:
        mask = np.abs(y_rounded - y_s) < 0.05
        if mask.sum() == 0:
            continue
        cx_s, cz_s, Cp_s = cx[mask], cz[mask], Cp[mask]
        x_le  = cx_s.min()
        chord = cx_s.max() - x_le
        if chord < 1e-6:
            continue
        xc_s = (cx_s - x_le) / chord

        up_m = cz_s > 0
        lo_m = ~up_m
        if up_m.sum() < 2 or lo_m.sum() < 2:
            continue

        for arr, sel in [(cp_up, up_m), (cp_lo, lo_m)]:
            xc_loc = xc_s[sel]
            cp_loc = Cp_s[sel]
            order  = np.argsort(xc_loc)
            arr   += w * np.interp(xc_common, xc_loc[order], cp_loc[order])
        w_sum += w

    if w_sum < 0.5:
        return None
    return xc_common, cp_up / w_sum, cp_lo / w_sum


y_target = HALF_SPAN / 2   # = 4.5

fig, ax = plt.subplots(figsize=(8, 5))

for res, mesh, marker, label in [
    (res_tri, mesh_tri, "o",  "Imported tri"),
    (res_par, mesh_par, "s",  "Parametric"),
]:
    result = _cp_at_y(
        mesh.centroids[:, 1], mesh.centroids[:, 0], mesh.centroids[:, 2],
        res["Cp"], y_target,
    )
    if result is None:
        print(f"   Warning: cannot interpolate Cp at y={y_target} for {label}")
        continue
    xc, cp_upper, cp_lower = result

    c = ax.plot(xc, cp_upper, marker + "-", markevery=8,
                label=f"{label} upper")[0].get_color()
    ax.plot(xc, cp_lower, marker + "--", color=c, markevery=8,
            label=f"{label} lower")

ax.invert_yaxis()
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("x/c")
ax.set_ylabel("Cp")
ax.set_title(f"Chordwise Cp at y = {y_target:.1f}  (α = {alpha_cp}°)")
ax.legend(fontsize=8)
ax.grid(True, linewidth=0.5)
plt.tight_layout()
fig.savefig(HERE / "07_Cp_wing.png", dpi=150)
plt.close(fig)
print("→ saved 07_Cp_wing.png")

# ---------------------------------------------------------------------------
# 6.  Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"{'Mesh':<20} {'Solver':<15} {'CL_α (1/rad)':>14}")
print(f"{'-'*60}")
print(f"{'Imported tri':<20} {'solve_morino':<15} {slope_tri:>14.4f}")
print(f"{'Parametric':<20} {'solve_morino':<15} {slope_par:>14.4f}")
print(f"{'-'*60}")
print(f"{'|error|':<20} {'':<15} {err_pct:>13.2f}%")
print(f"{'Target':<20} {'':<15} {'< 5.00%':>14}")
print(f"{'='*60}")
status = "✓ PASS" if err_pct < 5.0 else "✗ FAIL"
print(f"Stage 8 verification: {status}")

# ---------------------------------------------------------------------------
# 7.  Wing-body case: import wing portion only (body in Stage 9)
# ---------------------------------------------------------------------------
print("\n── Reading wing-body.tri …")
verts_wb, tris_wb, comp_ids_wb = read_tri(WBODY_TRI)
comp_names_wb = read_tkey(MESH_DIR / "wing-body.tkey")
print(f"   Components: {comp_names_wb}")

# Right wing only (comp 1)
mask_wb = comp_ids_wb == 1
mesh_wb = tris_to_mesh(
    verts_wb, tris_wb, comp_ids_wb,
    lifting_comp_ids=[1],
    mask=mask_wb,
)
print(f"   Wing-body wing: {mesh_wb.n_panels} panels, "
      f"{len(mesh_wb.te_pairs)} TE pairs")

# Build cam_mesh from the wing-body geometry (approximate — use S_ref from TE spans)
WB_HALF_SPAN = float(verts_wb[tris_wb[mask_wb].ravel(), 1].max())
WB_ROOT_CHORD = ROOT_CHORD     # approximate
WB_TIP_CHORD  = TIP_CHORD
WB_S_REF = 0.5 * (WB_ROOT_CHORD + WB_TIP_CHORD) * WB_HALF_SPAN
cam_mesh_wb = make_vlm_mesh(
    half_span=WB_HALF_SPAN, root_chord=WB_ROOT_CHORD, tip_chord=WB_TIP_CHORD,
    sweep_le=SWEEP_LE, n_span=len(mesh_wb.te_pairs), n_chord=10,
)
res_wb = solve_morino(mesh_wb, alpha_deg=5.0,
                      S_ref=WB_S_REF, b_ref=2 * WB_HALF_SPAN,
                      cam_mesh=cam_mesh_wb)
print(f"   Wing-body wing:  CL={res_wb['CL']:.4f}   CDi={res_wb['CDi']:.5f}")

print("\n✓ Stage 8 example complete.")
