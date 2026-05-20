"""Stage 5 demo — Prandtl-Glauert compressibility correction.

Uses the same swept flying wing as Stage 3 (AR ≈ 7.7, 35° sweep, taper 0.4).

Outputs (saved in the same directory as this script)
-----------------------------------------------------
04_CL_vs_mach.png   — CL(M) sweep at fixed α = 4°: VLM vs PG-corrected VLM
                       vs 1/β · CL_incomp reference curve.
04_CDi_vs_mach.png  — CDi(M) sweep: VLM vs PG-corrected VLM.
04_Cp_chordwise.png — Mid-span chordwise Cp at M = 0, 0.3, 0.5 (VLM ΔCp only).

Run from the repo root::

    python examples/04_compressibility/04_compressibility.py
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

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Wing geometry (same as Stage 3 swept flying wing) ─────────────────────────
HALF_SPAN  = 7.0
ROOT_CHORD = 3.5
TIP_CHORD  = 1.4       # taper ratio 0.4
SWEEP_LE   = 35.0
DIHEDRAL   = 0.0
TWIST_TIP  = -2.0
N_SPAN     = 16
N_CHORD    = 6
ALPHA      = 4.0       # fixed angle of attack for the Mach sweep

mesh = make_vlm_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, dihedral=DIHEDRAL, twist_tip=TWIST_TIP,
    n_span=N_SPAN, n_chord=N_CHORD,
)

# Physical reference quantities (from original mesh, before any PG transform)
S_ref = float(mesh.areas.sum())
b_ref = 2.0 * float(mesh.vertices[:, 1].max())
c_ref = S_ref / (0.5 * b_ref)
x_ref = 0.25 * c_ref

print(f"Wing: AR = {b_ref**2/S_ref:.2f}, S_ref = {S_ref:.3f} m², "
      f"b_ref = {b_ref:.3f} m, c_ref = {c_ref:.3f} m")

# ── Mach sweep ────────────────────────────────────────────────────────────────
machs = np.linspace(0.0, 0.75, 16)

r0 = solve_vlm(mesh, alpha_deg=ALPHA, mach=0.0,
               S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref)
CL0  = r0["CL"]
CDi0 = r0["CDi"]

print(f"\nIncompressible (M=0): CL={CL0:.4f}, CDi={CDi0:.5f}")
print(f"\nMach sweep at α = {ALPHA}°")

CL_pg   = []
CDi_pg  = []
betas   = []

for M in machs:
    r = solve_vlm(mesh, alpha_deg=ALPHA, mach=float(M),
                  S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref)
    CL_pg.append(r["CL"])
    CDi_pg.append(r["CDi"])
    betas.append(r["beta"])
    print(f"  M={M:.2f}  β={r['beta']:.4f}  CL={r['CL']:.4f}  "
          f"CL_ref={CL0/r['beta']:.4f}  CDi={r['CDi']:.5f}")

CL_pg   = np.array(CL_pg)
CDi_pg  = np.array(CDi_pg)
betas   = np.array(betas)

# ── Plot 1: CL vs Mach ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(machs, CL0 / betas,  "k--", lw=1.5,
        label=r"$C_{L,\mathrm{incomp}} / \beta$  (Glauert reference)")
ax.plot(machs, CL_pg, "b-o", ms=5, lw=1.8,
        label="VLM with PG correction (Stage 5)")
ax.axhline(CL0, color="gray", lw=1.0, ls=":", label=f"Incompressible  CL = {CL0:.3f}")

ax.set_xlabel("Mach number  M")
ax.set_ylabel(r"$C_L$")
ax.set_title(
    f"CL vs Mach — Swept flying wing  (AR={b_ref**2/S_ref:.1f}, "
    f"α={ALPHA}°, Λ={SWEEP_LE}°)\nPrandtl-Glauert correction"
)
ax.legend(fontsize=9)
ax.set_xlim(0, 0.8)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_CL_vs_mach.png"), dpi=150)
plt.close(fig)
print("\nSaved 04_CL_vs_mach.png")

# ── Plot 2: CDi vs Mach ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(machs, CDi0 / betas**2, "k--", lw=1.5,
        label=r"$C_{Di,\mathrm{incomp}} / \beta^2$  (Glauert reference)")
ax.plot(machs, CDi_pg, "r-s", ms=5, lw=1.8,
        label="VLM with PG correction (Stage 5)")
ax.axhline(CDi0, color="gray", lw=1.0, ls=":", label=f"Incompressible  CDi = {CDi0:.4f}")

ax.set_xlabel("Mach number  M")
ax.set_ylabel(r"$C_{Di}$")
ax.set_title(
    f"CDi vs Mach — Swept flying wing  (AR={b_ref**2/S_ref:.1f}, "
    f"α={ALPHA}°)\nPrandtl-Glauert correction"
)
ax.legend(fontsize=9)
ax.set_xlim(0, 0.8)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_CDi_vs_mach.png"), dpi=150)
plt.close(fig)
print("Saved 04_CDi_vs_mach.png")

# ── Plot 3: Chordwise Cp at three Mach numbers ────────────────────────────────
# The VLM returns per-panel ΔCp = dCL * S_ref / panel_area (not surface Cp).
# We use the ΔCp approach to show how lifting pressure difference grows with M.

plot_machs   = [0.0, 0.3, 0.5]
colors_upper = ["navy", "royalblue", "deepskyblue"]
colors_lower = ["darkred", "crimson", "salmon"]

js_mid = N_SPAN // 2   # mid-span strip index

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

print("\nChordwise ΔCp at mid-span:")
for ax, M_plot, cu, cl in zip(axes, plot_machs, colors_upper, colors_lower):
    r    = solve_vlm(mesh, alpha_deg=ALPHA, mach=M_plot,
                     S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref)
    beta = r["beta"]

    # ΔCp per panel = 2Γ / (V·Δx)
    A_v, B_v  = r["A"], r["B"]
    span_len  = np.maximum(np.linalg.norm(B_v - A_v, axis=-1), 1e-12)
    chord_len = mesh.areas / span_len
    dCp_all   = 2.0 * r["Gamma"] / (1.0 * chord_len)   # (Np,)

    # Extract mid-span strip: panels are ordered chord-major (ic*N_SPAN + js)
    x_pts  = np.array([mesh.centroids[ic * N_SPAN + js_mid, 0]
                        for ic in range(N_CHORD)])
    dCp_strip = np.array([dCp_all[ic * N_SPAN + js_mid] for ic in range(N_CHORD)])

    # Normalise x by local chord (approximate: use root chord for simplicity)
    xc = x_pts / ROOT_CHORD

    ax.plot(xc, -dCp_strip / 2, color=cu, lw=1.8,
            marker="o", ms=4, label="upper  (−ΔCp/2)")
    ax.plot(xc, +dCp_strip / 2, color=cl, lw=1.8,
            marker="s", ms=4, label="lower  (+ΔCp/2)")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_title(f"M = {M_plot:.1f}  (β = {beta:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    print(f"  M={M_plot:.1f}  ΔCp_max={dCp_strip.max():.3f}  "
          f"ΔCp_max_ref (×{1/beta:.3f}) = {dCp_strip.max()/beta:.3f}")

axes[0].set_ylabel("ΔCp / 2  (inverted — suction up)")
fig.suptitle(
    f"Mid-span chordwise ΔCp at α = {ALPHA}° — Prandtl-Glauert correction",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_Cp_chordwise.png"), dpi=150)
plt.close(fig)
print("Saved 04_Cp_chordwise.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== PG correction summary (α = {ALPHA}°) ===")
print(f"{'M':>5}  {'β':>6}  {'CL_PG':>8}  {'CL₀/β':>8}  {'err%':>7}  "
      f"{'CDi_PG':>9}  {'CDi₀/β²':>9}")
for M, b, cl, cdi in zip(machs[::3], betas[::3], CL_pg[::3], CDi_pg[::3]):
    ref_cl  = CL0 / b
    ref_cdi = CDi0 / b**2
    err     = (cl / ref_cl - 1.0) * 100 if ref_cl > 1e-6 else 0.0
    print(f"{M:5.2f}  {b:6.4f}  {cl:8.4f}  {ref_cl:8.4f}  {err:+7.2f}%  "
          f"{cdi:9.5f}  {ref_cdi:9.5f}")
