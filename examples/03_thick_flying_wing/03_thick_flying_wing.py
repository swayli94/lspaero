"""Stage 4 demo: thick-surface panel solver on a swept tapered flying wing.

Configuration
-------------
Span      : 10 m  (half-span = 5 m)
Root chord: 2.0 m
Tip chord : 0.6 m  (taper ratio 0.30)
LE sweep  : 30°
Dihedral  : 0°
Tip twist : −2°
Airfoil   : NACA 0012  (thick surface — upper + lower panels)
AR        : ≈ 6.25

Method
------
solve_morino combines two solves to produce accurate global forces and
physically correct surface Cp distributions:

  (A) Hess-Smith source solve on the thick surface at α = 0°
      → pure thickness Cp (symmetric, no lift).
      Solves AIC_σ · σ = −(V∞_0 · n̂) with V∞_0 along x-axis.
      Tangential surface velocity → Cp_thickness on each panel.
      Using α = 0° avoids embedding the stagnation-point shift into σ,
      which would double-count with the VLM lifting ΔCp.

  (B) VLM on the mean camber surface (vertex-averaged from upper/lower)
      → Kutta-Joukowski forces: CL, CDi, Cm  (primary output).
      The pressure difference from bound circulation:
          ΔCp_j = 2Γ_j / (V_∞ · Δx_j)    (thin-airfoil K-J)

  (C) Superimpose thickness and lifting effects; clip to Cp ≤ 1
      (Bernoulli hard bound in incompressible flow):
          Cp_upper = Cp_thickness − ΔCp / 2   (suction)
          Cp_lower = Cp_thickness + ΔCp / 2   (pressure, ≤ 1)

Outputs (saved in same directory as this script)
-------------------------------------------------
03_CL_vs_alpha.png         — CL(α): thick-panel vs VLM reference
03_Cp_contour.png          — Surface Cp contour on upper and lower wing
03_Cp_chordwise.png        — Chordwise Cp at three spanwise stations
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lspaero.geometry.wing import make_vlm_mesh, make_wing_mesh
from lspaero.solver.morino import solve_morino
from lspaero.solver.solve import solve_vlm

# ── Wing geometry ──────────────────────────────────────────────────────────
HALF_SPAN   = 5.0
ROOT_CHORD  = 2.0
TIP_CHORD   = 0.6
SWEEP_LE    = 30.0
DIHEDRAL    = 0.0
TWIST_TIP   = -2.0
AIRFOIL     = "0012"
N_SPAN      = 20
N_CHORD     = 10

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Meshes ─────────────────────────────────────────────────────────────────
thick_mesh = make_wing_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, dihedral=DIHEDRAL, twist_tip=TWIST_TIP,
    airfoil=AIRFOIL, n_span=N_SPAN, n_chord=N_CHORD,
)
vlm_mesh = make_vlm_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, dihedral=DIHEDRAL, twist_tip=TWIST_TIP,
    airfoil=AIRFOIL, n_span=N_SPAN, n_chord=N_CHORD,
)

n_up   = int((thick_mesh.surface_id == 0).sum())
n_span = N_SPAN
n_chord = N_CHORD

# ── CL vs alpha sweep ──────────────────────────────────────────────────────
alphas = np.linspace(-4, 12, 9)
CL_thick = []
CL_vlm   = []

print("Sweeping alpha...")
for alpha in alphas:
    r = solve_morino(thick_mesh, alpha_deg=alpha)
    v = solve_vlm(vlm_mesh,     alpha_deg=alpha)
    CL_thick.append(r["CL"])
    CL_vlm.append(v["CL"])
    print(f"  α={alpha:5.1f}°  CL_thick={r['CL']:.4f}  CL_vlm={v['CL']:.4f}")

# ── Design-point Cp ───────────────────────────────────────────────────────
ALPHA_DESIGN = 5.0
result = solve_morino(thick_mesh, alpha_deg=ALPHA_DESIGN)
Cp     = result["Cp"]           # (Np,)  upper panels first

# ── Plot 1: CL vs alpha ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, CL_vlm,   "k--", label="VLM (camber surface)", linewidth=1.5)
ax.plot(alphas, CL_thick, "b-o", label="Thick panel (Stage 4)", markersize=5)
ax.set_xlabel("Angle of attack α (°)")
ax.set_ylabel("Lift coefficient CL")
ax.set_title("CL vs α — swept flying wing (NACA 0012, AR ≈ 6.25)")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_CL_vs_alpha.png"), dpi=150)
plt.close(fig)
print("Saved 03_CL_vs_alpha.png")

# ── Plot 2: Surface Cp contour ─────────────────────────────────────────────
pv_up = thick_mesh.vertices[thick_mesh.panels[thick_mesh.surface_id == 0]]   # (n_up, 4, 3)
pv_lo = thick_mesh.vertices[thick_mesh.panels[thick_mesh.surface_id == 1]]   # (n_lo, 4, 3)

Cp_up = Cp[:n_up]
Cp_lo = Cp[n_up:n_up + len(pv_lo)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, pv, Cp_surf, title in [
    (axes[0], pv_up, Cp_up, "Upper surface"),
    (axes[1], pv_lo, Cp_lo, "Lower surface"),
]:
    # Centroid x, y and Cp per panel for scatter plot
    cx = pv[:, :, 0].mean(axis=1)   # x-centroid (chordwise)
    cy = pv[:, :, 1].mean(axis=1)   # y-centroid (spanwise)
    # Extend to full span (symmetric image)
    cx_full = np.concatenate([cx, cx])
    cy_full = np.concatenate([-cy, cy])
    Cp_full = np.concatenate([Cp_surf, Cp_surf])

    vmin, vmax = -1.5, 1.5
    sc = ax.scatter(cy_full, cx_full, c=Cp_full, cmap="RdBu_r",
                    vmin=vmin, vmax=vmax, s=8, linewidths=0)
    ax.set_xlabel("Spanwise y (m)")
    ax.set_ylabel("Chordwise x (m)")
    ax.set_title(f"Cp — {title}  (α = {ALPHA_DESIGN}°)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Cp")

fig.suptitle("NACA 0012 swept wing  —  thick-panel surface Cp", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_Cp_contour.png"), dpi=150)
plt.close(fig)
print("Saved 03_Cp_contour.png")

# ── Plot 3: Chordwise Cp at three spanwise stations ────────────────────────
js_list   = [0, N_SPAN // 2, N_SPAN - 2]
js_labels = ["Root (js=0)", f"Mid-span (js={N_SPAN//2})", f"Near-tip (js={N_SPAN-2})"]
colors    = ["b", "g", "r"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, js, label, color in zip(axes, js_list, js_labels, colors):
    x_vals   = []
    Cp_upper = []
    Cp_lower = []
    for ic in range(n_chord):
        up_idx = ic * n_span + js
        lo_idx = n_up + up_idx
        x = thick_mesh.centroids[up_idx, 0]
        x_vals.append(x)
        Cp_upper.append(Cp[up_idx])
        Cp_lower.append(Cp[lo_idx])

    ax.plot(x_vals, Cp_upper, f"{color}-o", label="Upper", markersize=4)
    ax.plot(x_vals, Cp_lower, f"{color}--s", label="Lower", markersize=4)
    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Chordwise x (m)")
    ax.set_title(f"{label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

axes[0].set_ylabel("Cp  (inverted axis)")
fig.suptitle(
    f"Chordwise Cp distribution  —  NACA 0012, α = {ALPHA_DESIGN}°  "
    f"(CL = {result['CL']:.3f})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_Cp_chordwise.png"), dpi=150)
plt.close(fig)
print("Saved 03_Cp_chordwise.png")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n=== Design point α = {ALPHA_DESIGN}° ===")
print(f"  CL   (VLM K-J)     = {result['CL']:.4f}")
print(f"  CDi  (VLM K-J)     = {result['CDi']:.5f}")
print(f"  Cm   (VLM K-J)     = {result['Cm']:.4f}")
print(f"  CL   (Cp integral) = {result['CL_cp']:.4f}")
print(f"  Cp_upper min/max   = {Cp[:n_up].min():.3f} / {Cp[:n_up].max():.3f}")
print(f"  Cp_lower min/max   = {Cp[n_up:].min():.3f} / {Cp[n_up:].max():.3f}")
