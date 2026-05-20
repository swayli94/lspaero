"""Example 08 — Wing-body combination.

Demonstrates Stage 9: a swept flying wing combined with a Sears–Haack fuselage.

The solver uses:
  • Hess-Smith source panels on the entire surface (wing + body)
  • VLM circulation only on the wing (cam_mesh)
  • No Kutta condition or wake for body panels (lifting_panels = False)

Outputs
-------
08_CL_vs_alpha.png   : CL(α) for wing-alone vs wing+body
08_Cp_wing.png       : chordwise Cp at three spanwise stations (with and without body)
08_Cp_body.png       : axial Cp distribution on the body at ϕ = 0°, 90°, 180°

Usage
-----
    python examples/08_wing_body/08_wing_body.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI/headless runs
import matplotlib.pyplot as plt

# --- project imports ---
from lspaero.geometry.wing  import make_wing_mesh, make_vlm_mesh
from lspaero.geometry.body  import make_body_mesh, sears_haack_profile
from lspaero.geometry.mesh  import combine_meshes
from lspaero.solver.morino  import solve_morino

# ============================================================
# Geometry parameters
# ============================================================

# Wing — swept flying-wing configuration
HALF_SPAN   = 5.0
ROOT_CHORD  = 2.0
TIP_CHORD   = 1.0
SWEEP_LE    = 30.0     # degrees LE sweep
AIRFOIL     = "0012"
N_SPAN      = 12
N_CHORD     = 8

# Body — Sears–Haack fuselage
BODY_LENGTH = ROOT_CHORD * 1.6   # 3.2 — longer than wing root chord
BODY_R_MAX  = ROOT_CHORD * 0.07  # 0.14 — slender, r/L ≈ 4.4%
BODY_X_OFFSET = -ROOT_CHORD * 0.3  # nose starts forward of LE
N_AXIAL     = 20
N_CIRC      = 16

# Flow conditions
RHO         = 1.225
V_MAG       = 1.0
ALPHAS_DEG  = np.arange(-4, 13, 2, dtype=float)   # alpha sweep

# Reference quantities (from physical wing)
wing_ref = make_wing_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, airfoil=AIRFOIL, n_span=N_SPAN, n_chord=N_CHORD,
)
S_REF = float(wing_ref.areas[wing_ref.surface_id == 0].sum())
B_REF = 2.0 * HALF_SPAN
C_REF = S_REF / (0.5 * B_REF)
X_REF = 0.25 * C_REF
print(f"Wing S_ref={S_REF:.4f}  b_ref={B_REF:.4f}  c_ref={C_REF:.4f}")

# ============================================================
# Build meshes
# ============================================================
print("\nBuilding meshes …")

wing_mesh = make_wing_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, airfoil=AIRFOIL, n_span=N_SPAN, n_chord=N_CHORD,
)
vlm_mesh = make_vlm_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, airfoil=AIRFOIL, n_span=N_SPAN, n_chord=N_CHORD,
)

body_profile = sears_haack_profile(length=BODY_LENGTH, r_max=BODY_R_MAX)
body_mesh = make_body_mesh(
    length=BODY_LENGTH,
    r_profile=body_profile,
    n_axial=N_AXIAL,
    n_circ=N_CIRC,
    x_offset=BODY_X_OFFSET,
)
combined_mesh = combine_meshes([wing_mesh, body_mesh])

print(f"Wing panels    : {wing_mesh.n_panels}")
print(f"Body panels    : {body_mesh.n_panels}")
print(f"Combined panels: {combined_mesh.n_panels}")

# ============================================================
# α sweep: wing-alone vs wing + body
# ============================================================
print("\nRunning α sweep …")

ref_kw = dict(S_ref=S_REF, b_ref=B_REF, c_ref=C_REF, x_ref=X_REF,
              V_mag=V_MAG, rho=RHO)

CL_wing = []
CDi_wing = []
Cm_wing = []

CL_wb = []
CDi_wb = []
Cm_wb = []

for alpha in ALPHAS_DEG:
    r_w = solve_morino(wing_mesh, alpha_deg=float(alpha), **ref_kw)
    CL_wing.append(r_w["CL"])
    CDi_wing.append(r_w["CDi"])
    Cm_wing.append(r_w["Cm"])

    r_wb = solve_morino(combined_mesh, cam_mesh=vlm_mesh,
                        alpha_deg=float(alpha), **ref_kw)
    CL_wb.append(r_wb["CL"])
    CDi_wb.append(r_wb["CDi"])
    Cm_wb.append(r_wb["Cm"])

CL_wing  = np.array(CL_wing)
CDi_wing = np.array(CDi_wing)
Cm_wing  = np.array(Cm_wing)
CL_wb    = np.array(CL_wb)
CDi_wb   = np.array(CDi_wb)
Cm_wb    = np.array(Cm_wb)

# Lift-curve slopes
valid = (ALPHAS_DEG >= 0) & (ALPHAS_DEG <= 8)
p_w  = np.polyfit(np.radians(ALPHAS_DEG[valid]), CL_wing[valid],  1)
p_wb = np.polyfit(np.radians(ALPHAS_DEG[valid]), CL_wb[valid],    1)
CLa_wing = p_w[0];   CLa_wb = p_wb[0]
print(f"\nCL_α wing-alone : {CLa_wing:.4f} /rad")
print(f"CL_α wing+body  : {CLa_wb:.4f} /rad")
print(f"Body ΔCL_α (VLM): {CLa_wb - CLa_wing:+.5f} /rad")
print(
    "  Note: primary CL from VLM K-J is decoupled from body sources.\n"
    "  Body interference appears in the Cp distribution (see 08_Cp_wing.png).\n"
    "  A fully coupled formulation is needed to capture the ~1% CL increment\n"
    "  from body-wing source interference."
)

# ============================================================
# Detailed solve at α = 5° for Cp plots
# ============================================================
ALPHA_CP = 5.0
print(f"\nSolving at α = {ALPHA_CP}° for Cp plots …")

r_w5  = solve_morino(wing_mesh, alpha_deg=ALPHA_CP, **ref_kw)
r_wb5 = solve_morino(combined_mesh, cam_mesh=vlm_mesh,
                     alpha_deg=ALPHA_CP, **ref_kw)

Cp_wing_alone  = r_w5["Cp"]
Cp_wing_body   = r_wb5["Cp"][:wing_mesh.n_panels]   # wing portion only
Cp_body        = r_wb5["Cp"][wing_mesh.n_panels:]    # body portion

# ============================================================
# Plot 1 — CL(α)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(ALPHAS_DEG, CL_wing, "b-o", ms=4, label=f"Wing alone  (CLα={CLa_wing:.2f}/rad)")
ax.plot(ALPHAS_DEG, CL_wb,   "r-s", ms=4, label=f"Wing+body   (CLα={CLa_wb:.2f}/rad)")
ax.set_xlabel("α  (deg)"); ax.set_ylabel("CL")
ax.set_title("CL vs α"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

ax = axes[1]
ax.plot(ALPHAS_DEG, CDi_wing, "b-o", ms=4, label="Wing alone")
ax.plot(ALPHAS_DEG, CDi_wb,   "r-s", ms=4, label="Wing+body")
ax.set_xlabel("α  (deg)"); ax.set_ylabel("CDi")
ax.set_title("CDi vs α"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

ax = axes[2]
ax.plot(ALPHAS_DEG, Cm_wing, "b-o", ms=4, label="Wing alone")
ax.plot(ALPHAS_DEG, Cm_wb,   "r-s", ms=4, label="Wing+body")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("α  (deg)"); ax.set_ylabel("Cm")
ax.set_title("Cm vs α"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

fig.suptitle(
    f"Stage 9 — Wing-body combination  "
    f"(AR≈{B_REF**2/S_REF/2:.1f}, Λ={SWEEP_LE}°, NACA {AIRFOIL}; "
    f"Sears–Haack body L/D={BODY_LENGTH/BODY_R_MAX:.0f})",
    fontsize=10
)
fig.tight_layout()
out1 = "examples/08_wing_body/08_CL_vs_alpha.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}")

# ============================================================
# Plot 2 — Chordwise Cp at three spanwise stations
# ============================================================
# Spanwise stations: near root (~5%), mid (~35%), tip (~70%)
# Near-root station captures the body-wing interference Cp effect.
y_stations = [HALF_SPAN * 0.05, HALF_SPAN * 0.35, HALF_SPAN * 0.70]
colors = ["C0", "C1", "C2"]

upper_mask_w = wing_mesh.surface_id == 0
lower_mask_w = wing_mesh.surface_id == 1
cent_w = wing_mesh.centroids

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

for idx_s, (y_tgt, col) in enumerate(zip(y_stations, colors)):
    ax = axes[idx_s]

    # Wing-alone Cp
    for (mask, sign, ls_base) in [(upper_mask_w, -1, "b"), (lower_mask_w, +1, "r")]:
        dist_y = np.abs(cent_w[mask, 1] - y_tgt)
        j_strip = np.argmin(dist_y.reshape(N_CHORD, N_SPAN), axis=1)
        # j_strip has one value per chordwise row; use the closest spanwise column
        flat_idx = np.arange(N_CHORD) * N_SPAN + j_strip
        panels_in_strip = np.where(mask)[0][flat_idx]
        x_c = cent_w[panels_in_strip, 0]
        cp_w  = Cp_wing_alone[panels_in_strip]
        cp_wb = Cp_wing_body[panels_in_strip]

        label_type = "upper" if sign == -1 else "lower"
        ax.plot(x_c, cp_w,  color=ls_base, lw=1.2, ls="-",
                label=f"{label_type} alone" if idx_s == 0 else None)
        ax.plot(x_c, cp_wb, color=ls_base, lw=1.2, ls="--",
                label=f"{label_type} +body" if idx_s == 0 else None)

    ax.invert_yaxis()
    ax.set_xlabel("x  (m)")
    y_rel = y_tgt / HALF_SPAN
    label = f"y/(b/2) = {y_rel:.2f}"
    if y_rel < 0.10:
        label += "  (root — interference)"
    ax.set_title(label)
    if idx_s == 0:
        ax.set_ylabel("Cp  (inverted)")
    ax.grid(True, alpha=0.4)

axes[0].legend(fontsize=8, ncol=2)
fig.suptitle(
    f"Chordwise Cp at α = {ALPHA_CP}°  (solid = wing-alone, dashed = wing+body)\n"
    f"Body interference visible at root (max |ΔCp| ≈ "
    f"{np.abs(Cp_wing_body - Cp_wing_alone).max():.2f}; "
    f"source coupling from combined Hess-Smith solve)",
    fontsize=9
)
fig.tight_layout()
out2 = "examples/08_wing_body/08_Cp_wing.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved {out2}")

# ============================================================
# Plot 3 — Axial Cp distribution on the body
# ============================================================
# Three circumferential angles: ϕ = 0° (starboard), 90° (top), 180° (port)
#   y = r·cos(θ),  z = r·sin(θ)  →  ϕ=0° → y>0, z=0
#                                     ϕ=90° → y=0, z>0
#                                     ϕ=180° → y<0, z=0

body_cent = body_mesh.centroids
body_y = body_cent[:, 1]
body_z = body_cent[:, 2]
body_x = body_cent[:, 0]
body_theta = np.arctan2(body_z, body_y)   # angle in y-z plane

phi_targets = [0.0, np.pi / 2, np.pi]
phi_labels  = ["ϕ = 0° (starboard)", "ϕ = 90° (top)", "ϕ = 180° (port)"]
phi_colors  = ["C0", "C1", "C2"]
phi_tol     = np.pi / N_CIRC + 0.01   # half-panel angular width + small margin

fig, ax = plt.subplots(figsize=(8, 4.5))

for phi, label, col in zip(phi_targets, phi_labels, phi_colors):
    # Select panels near this circumferential angle
    dtheta = np.abs(np.arctan2(np.sin(body_theta - phi), np.cos(body_theta - phi)))
    strip_mask = dtheta < phi_tol
    if not strip_mask.any():
        print(f"  WARNING: no panels found near ϕ={np.degrees(phi):.0f}°")
        continue
    sort_idx = np.argsort(body_x[strip_mask])
    x_strip  = body_x[strip_mask][sort_idx]
    cp_strip = Cp_body[strip_mask][sort_idx]
    ax.plot(x_strip, cp_strip, color=col, marker=".", ms=4, lw=1.2, label=label)

ax.invert_yaxis()
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("x  (m)"); ax.set_ylabel("Cp  (inverted)")
ax.set_title(
    f"Body surface Cp at α = {ALPHA_CP}°  "
    f"(Sears–Haack  L={BODY_LENGTH:.2f} m,  r_max={BODY_R_MAX:.3f} m)"
)
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
fig.tight_layout()
out3 = "examples/08_wing_body/08_Cp_body.png"
fig.savefig(out3, dpi=150)
plt.close(fig)
print(f"Saved {out3}")

# ============================================================
# Summary table
# ============================================================
print("\n" + "="*63)
print(f"{'Quantity':<26} {'Wing-alone':>16} {'Wing+body':>16}")
print("="*63)
a_ref = 5
r_w_ref  = solve_morino(wing_mesh,     alpha_deg=a_ref, **ref_kw)
r_wb_ref = solve_morino(combined_mesh, cam_mesh=vlm_mesh, alpha_deg=a_ref, **ref_kw)
for key in ["CL", "CDi", "Cm", "CL_cp", "CDi_cp"]:
    print(f"  {key:<24} {r_w_ref[key]:>16.5f} {r_wb_ref[key]:>16.5f}")
print(f"  {'L/D':<24} {r_w_ref['CL']/max(r_w_ref['CDi'],1e-10):>16.2f} "
      f"{r_wb_ref['CL']/max(r_wb_ref['CDi'],1e-10):>16.2f}")
print(f"  {'CL_α VLM (0–8°)':<24} {CLa_wing:>16.4f} {CLa_wb:>16.4f}")
print(f"  {'max |ΔCp| body–wing':<24} {'—':>16} "
      f"{np.abs(r_wb_ref['Cp'][:wing_mesh.n_panels] - r_w_ref['Cp']).max():>16.4f}")
print("="*63)
print(
    "\n  CL (VLM K-J) is identical because the VLM is decoupled from body\n"
    "  sources.  Body interference is captured in the Cp distribution and\n"
    "  in CL_cp (Cp-integrated), which shows the source-coupling effect."
)
print("\nAll plots saved to examples/08_wing_body/")
