"""Stage 4 demo: thick-panel solve_morino vs 2D Hess-Smith on NACA 0012.

A high-aspect-ratio (AR = 100) rectangular wing approximates 2D flow.  The
mid-span chordwise Cp strip from solve_morino is compared with the 2D
Hess-Smith reference from Stage 1.

Method notes
------------
* A 3D Hess-Smith source solve on the thick surface gives thickness Cp on both
  the upper and lower surface panels, including the suction peak at α = 0°.
* VLM on the mean camber surface gives Kutta-Joukowski forces (CL, CDi, Cm)
  and the lifting pressure difference ΔCp = 2Γ/(V∞ Δx) per panel.
* The two effects are superimposed: Cp_upper = Cp_thickness − ΔCp/2,
  Cp_lower = Cp_thickness + ΔCp/2.

Outputs (saved in same directory as this script)
-------------------------------------------------
00b_CL_vs_alpha.png    — CL(α): thin-airfoil theory, Hess-Smith, thick-panel
00b_Cp_chordwise.png   — Chordwise Cp at α = 0°, 4°, 8°
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lspaero.geometry.naca import naca4
from lspaero.geometry.wing import make_wing_mesh
from lspaero.solver.morino import solve_morino

# Load Stage-1 helpers (file name begins with a digit, so use importlib)
_spec = importlib.util.spec_from_file_location(
    "stage1_hs", os.path.join(os.path.dirname(__file__), "00_2d_airfoil.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
solve_hess_smith   = _mod.solve_hess_smith
_panel_geometry    = _mod._panel_geometry
_split_upper_lower = _mod._split_upper_lower

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Quasi-2D wing geometry ─────────────────────────────────────────────────
AIRFOIL   = "0012"
CHORD     = 1.0
N_CHORD   = 20     # chordwise panels per surface
N_SPAN    = 5      # spanwise panels (low count — only mid-span strip needed)
HALF_SPAN = 50.0   # AR = 2 * 50 / 1 = 100  (≈ 2-D limit)

thick_mesh = make_wing_mesh(
    half_span=HALF_SPAN, root_chord=CHORD, tip_chord=CHORD,
    sweep_le=0.0, dihedral=0.0, twist_tip=0.0,
    airfoil=AIRFOIL, n_span=N_SPAN, n_chord=N_CHORD,
)
n_up   = int((thick_mesh.surface_id == 0).sum())   # = N_CHORD * N_SPAN
js_mid = N_SPAN // 2                                # mid-span strip index

# ── 2D Hess-Smith mesh (unit chord, 100 panels) ────────────────────────────
N_HS = 100
x_2d, z_2d = naca4(AIRFOIL, N_HS)
xm_2d, *_rest = _panel_geometry(x_2d, z_2d)
n_half = N_HS // 2

# ── CL vs alpha sweep ──────────────────────────────────────────────────────
alphas    = np.linspace(-4, 12, 9)
CL_theory = 2.0 * np.pi * np.radians(alphas)
CL_hs     = []
CL_tp     = []

print("CL vs alpha sweep ...")
for alpha in alphas:
    _, _, _, CL = solve_hess_smith(x_2d, z_2d, alpha)
    CL_hs.append(CL)
    r = solve_morino(thick_mesh, alpha_deg=alpha)
    CL_tp.append(r["CL"])
    print(f"  α={alpha:+6.1f}°  HS={CL:.4f}  TP={r['CL']:.4f}"
          f"  theory={2*np.pi*np.radians(alpha):.4f}")

CL_hs = np.array(CL_hs)
CL_tp = np.array(CL_tp)

# ── Plot 1: CL vs alpha ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, CL_theory, "k--",  lw=1.5, label=r"$2\pi\alpha$ (thin-airfoil theory)")
ax.plot(alphas, CL_hs,     "r-o",  ms=5,   label="2D Hess-Smith (Stage 1)")
ax.plot(alphas, CL_tp,     "b-s",  ms=5,   label=f"Thick-panel VLM (Stage 4, AR={int(2*HALF_SPAN/CHORD)})")
ax.set_xlabel("Angle of attack α (°)")
ax.set_ylabel("Lift coefficient CL")
ax.set_title("NACA 0012 — CL vs α  (2D Hess-Smith vs thick-panel quasi-2D)")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "00b_CL_vs_alpha.png"), dpi=150)
plt.close(fig)
print("Saved 00b_CL_vs_alpha.png")

# ── Plot 2: Chordwise Cp comparison ───────────────────────────────────────
cp_alphas = [0.0, 4.0, 8.0]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

print("\nChordwise Cp comparison ...")
for ax, alpha in zip(axes, cp_alphas):
    # 2D Hess-Smith
    _, _, Cp_hs, _ = solve_hess_smith(x_2d, z_2d, alpha)
    xl_hs, Cpl_hs, xu_hs, Cpu_hs = _split_upper_lower(xm_2d, Cp_hs, N_HS)

    # Thick-panel: mid-span strip
    r      = solve_morino(thick_mesh, alpha_deg=alpha)
    Cp_all = r["Cp"]

    x_up  = np.array([thick_mesh.centroids[ic * N_SPAN + js_mid, 0] for ic in range(N_CHORD)])
    Cp_up = np.array([Cp_all[ic * N_SPAN + js_mid]                  for ic in range(N_CHORD)])
    Cp_lo = np.array([Cp_all[n_up + ic * N_SPAN + js_mid]           for ic in range(N_CHORD)])

    ax.plot(xu_hs, Cpu_hs, "b-",    lw=1.5, label="HS upper")
    ax.plot(xl_hs, Cpl_hs, "r-",    lw=1.5, label="HS lower")
    ax.plot(x_up,  Cp_up,  "b--o",  lw=1.2, ms=4, label="TP upper")
    ax.plot(x_up,  Cp_lo,  "r--s",  lw=1.2, ms=4, label="TP lower")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_title(f"α = {alpha}°")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    print(f"  α={alpha}°  TP Cp_upper: {Cp_up.min():.3f} .. {Cp_up.max():.3f}"
          f"  HS Cp_upper: {Cpu_hs.min():.3f} .. {Cpu_hs.max():.3f}")

axes[0].set_ylabel("Cp  (inverted, suction up)")
fig.suptitle(
    f"NACA 0012 — Thick-panel quasi-2D (AR={int(2*HALF_SPAN/CHORD)}) vs 2D Hess-Smith",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "00b_Cp_chordwise.png"), dpi=150)
plt.close(fig)
print("Saved 00b_Cp_chordwise.png")

# ── CL slope summary ───────────────────────────────────────────────────────
mask     = (alphas >= -4.0) & (alphas <= 8.0)
slope_hs = np.polyfit(np.radians(alphas[mask]), CL_hs[mask], 1)[0]
slope_tp = np.polyfit(np.radians(alphas[mask]), CL_tp[mask], 1)[0]
print(f"\n=== CL slope summary ===")
print(f"  Thin-airfoil theory : {2*np.pi:.4f} /rad")
print(f"  2D Hess-Smith       : {slope_hs:.4f} /rad  ({(slope_hs/( 2*np.pi)-1)*100:+.2f}%)")
print(f"  Thick-panel (AR=100): {slope_tp:.4f} /rad  ({(slope_tp/(2*np.pi)-1)*100:+.2f}%)")
