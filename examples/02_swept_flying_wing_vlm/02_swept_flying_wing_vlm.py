"""Stage 3 demo: VLM on a swept tapered flying-wing configuration.

Configuration
-------------
Span      : 10 m  (half-span = 5 m)
Root chord: 2.0 m
Tip chord : 0.6 m  (taper ratio 0.30)
LE sweep  : 30°
Dihedral  : 0°
Tip twist : −2°  (washout, stabilises spanwise loading)
Airfoil   : NACA 0012 camber surface (thin, symmetric → CL0 = 0 at α = 0)
AR        : ≈ 6.25  (b² / S_plan)

Outputs (saved in same directory as this script)
-------------------------------------------------
02_CL_Cm_vs_alpha.png  — CL and Cm vs alpha
02_CL_CDi_polar.png    — Drag polar with elliptic reference
02_spanwise_loading.png — Spanwise CL distribution at trim-ish alpha
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lspaero.geometry.wing import make_vlm_mesh
from lspaero.solver.solve import solve_vlm

HERE = os.path.dirname(__file__)

# --------------------------------------------------------------------------- #
# Wing geometry                                                                #
# --------------------------------------------------------------------------- #
HALF_SPAN  = 5.0
ROOT_CHORD = 2.0
TIP_CHORD  = 0.6
SWEEP_LE   = 30.0   # degrees
TWIST_TIP  = -2.0   # degrees (wash-out)
N_SPAN     = 16
N_CHORD    = 4

# Planform reference quantities (used consistently below)
b_full   = 2 * HALF_SPAN
S_plan   = 0.5 * (ROOT_CHORD + TIP_CHORD) * b_full
AR_plan  = b_full**2 / S_plan
MAC      = (2/3) * ROOT_CHORD * (1 + 0.3 + 0.3**2) / (1 + 0.3)  # trapezoidal MAC
print(f"Flying-wing configuration:")
print(f"  Span      : {b_full:.1f} m")
print(f"  S_plan    : {S_plan:.2f} m²")
print(f"  AR        : {AR_plan:.2f}")
print(f"  MAC       : {MAC:.3f} m")
print(f"  LE sweep  : {SWEEP_LE:.0f}°")
print(f"  Tip twist : {TWIST_TIP:.0f}°  (washout)")
print()

mesh = make_vlm_mesh(
    half_span=HALF_SPAN,
    root_chord=ROOT_CHORD,
    tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE,
    twist_tip=TWIST_TIP,
    n_span=N_SPAN,
    n_chord=N_CHORD,
)

# --------------------------------------------------------------------------- #
# 1.  Alpha sweep: CL and Cm                                                  #
# --------------------------------------------------------------------------- #
alphas = np.linspace(-6, 18, 25)
results = [solve_vlm(mesh, alpha_deg=a) for a in alphas]
CLs  = [r["CL"]  for r in results]
CDis = [r["CDi"] for r in results]
Cms  = [r["Cm"]  for r in results]

# CL_alpha and zero-lift alpha by linear fit
CL_arr  = np.array(CLs)
alp_arr = np.radians(alphas)
coeffs  = np.polyfit(alp_arr, CL_arr, 1)   # linear fit
CL_alpha_vlm = coeffs[0]
alpha0_deg   = np.degrees(-coeffs[1] / coeffs[0])

# Cm slope
Cm_arr   = np.array(Cms)
coeffs_m = np.polyfit(alp_arr, Cm_arr, 1)
Cm_alpha_vlm = coeffs_m[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(alphas, CLs, "b-o", ms=4, label=f"CL  (slope={CL_alpha_vlm:.3f} /rad)")
ax.plot(alphas, Cms, "r-s", ms=4, label=f"Cm  (slope={Cm_alpha_vlm:.3f} /rad)")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("α  (deg)")
ax.set_ylabel("CL  /  Cm")
ax.set_title("Swept flying-wing: CL and Cm vs α\n"
             f"AR={AR_plan:.2f}, Λ={SWEEP_LE}°, twist={TWIST_TIP}°")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Annotate zero-lift alpha
ax.annotate(f"α₀ = {alpha0_deg:.2f}°",
            xy=(alpha0_deg, 0), xytext=(alpha0_deg + 2, 0.2),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

CDi_arr = np.array(CDis)
ax2 = axes[1]
ax2.plot(CL_arr, CDi_arr, "b-o", ms=4, label="VLM (near-field K-J)")
cdi_elliptic = CL_arr**2 / (np.pi * AR_plan)
ax2.plot(CL_arr, cdi_elliptic, "k--", lw=1.5,
         label=f"Elliptic: CL²/(π·AR)")
ax2.set_xlabel("CL")
ax2.set_ylabel("CDi")
ax2.set_title("Drag polar")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "02_CL_Cm_vs_alpha.png"), dpi=150)
plt.close(fig)
print("Saved 02_CL_Cm_vs_alpha.png")


# --------------------------------------------------------------------------- #
# 2.  L/D vs CL                                                               #
# --------------------------------------------------------------------------- #
V_mag = 50.0   # m/s for dimensional drag (rho=1.225)
rho   = 1.225
results_dim = [solve_vlm(mesh, alpha_deg=a, V_mag=V_mag, rho=rho,
                          S_ref=S_plan) for a in alphas]
CLs_d  = [r["CL"]  for r in results_dim]
CDis_d = [r["CDi"] for r in results_dim]

LoD = np.array(CLs_d) / np.maximum(np.array(CDis_d), 1e-9)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(CLs_d, LoD, "b-o", ms=4)
ax.set_xlabel("CL")
ax.set_ylabel("CL / CDi  (inviscid L/D)")
ax.set_title("Inviscid L/D  (induced drag only)")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "02_CL_CDi_polar.png"), dpi=150)
plt.close(fig)
print("Saved 02_CL_CDi_polar.png")


# --------------------------------------------------------------------------- #
# 3.  Spanwise loading at a representative alpha                               #
# --------------------------------------------------------------------------- #
alpha_load = 8.0
# Use n_span=8 for the loading plot.  Higher panel counts make the outermost
# trailing vortex (exactly at y=b/2) dangerously close to the tip-strip
# collocation points; large off-diagonal AIC terms then cause a spurious
# spike in Γ at the last strip.  n_span=8 has monotone, artefact-free loading.
mesh_load = make_vlm_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, twist_tip=TWIST_TIP,
    n_span=8, n_chord=N_CHORD,
)
res_load  = solve_vlm(mesh_load, alpha_deg=alpha_load)

A   = res_load["A"]
B   = res_load["B"]
dCL = res_load["dCL"]

y_mid    = 0.5 * (A[:, 1] + B[:, 1])
dy_panel = B[:, 1] - A[:, 1]
CL_load  = res_load["CL"]

# Sum chordwise panels per spanwise strip, then normalise by Δη to get dCL/dη.
unique_y  = np.unique(np.round(y_mid, 6))
dCL_strip = np.array([dCL[np.round(y_mid, 6) == yy].sum()      for yy in unique_y])
dy_strip  = np.array([dy_panel[np.round(y_mid, 6) == yy].mean() for yy in unique_y])
eta_strip = unique_y / HALF_SPAN
deta      = dy_strip / HALF_SPAN
loading   = dCL_strip / deta   # ∝ Γ(y), peaks at root

# Elliptic reference matched at root
elliptic_shape = np.sqrt(np.maximum(1 - eta_strip**2, 0))
elliptic_ref   = elliptic_shape * (loading[0] / max(elliptic_shape[0], 1e-12))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(eta_strip, loading,      "b-o", ms=4,
        label=f"VLM, α={alpha_load}°, CL={CL_load:.3f}")
ax.plot(eta_strip, elliptic_ref, "k--", lw=1.5, label="Elliptic ref")
ax.set_xlabel("η = y / (b/2)")
ax.set_ylabel("Spanwise loading  dCL/dη")
ax.set_title(f"Spanwise loading  —  swept flying wing, α={alpha_load}°")
ax.set_xlim(0, 1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "02_spanwise_loading.png"), dpi=150)
plt.close(fig)
print("Saved 02_spanwise_loading.png")


# --------------------------------------------------------------------------- #
# Summary printout                                                              #
# --------------------------------------------------------------------------- #
alpha_ref = 6.0
r5 = solve_vlm(mesh, alpha_deg=alpha_ref)
cla = r5["CL"] / np.radians(alpha_ref)

print()
print("=" * 55)
print(f"  Swept flying wing  AR = {AR_plan:.2f}")
print(f"  n_span={N_SPAN}, n_chord={N_CHORD}")
print("=" * 55)
print(f"  CL_alpha  : {CL_alpha_vlm:.4f} /rad  (linear fit)")
print(f"  alpha_0   : {alpha0_deg:.2f} deg  (zero-lift angle)")
print(f"  Cm_alpha  : {Cm_alpha_vlm:.4f} /rad  ({'stable' if Cm_alpha_vlm < 0 else 'unstable'})")
print(f"  CL  at α={alpha_ref}° : {r5['CL']:.4f}")
print(f"  CDi at α={alpha_ref}° : {r5['CDi']:.5f}")
print(f"  Cm  at α={alpha_ref}° : {r5['Cm']:.5f}")
print("=" * 55)
