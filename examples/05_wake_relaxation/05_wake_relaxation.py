"""Stage 6 demo — free-wake relaxation for VLM.

Uses an AR ≈ 6.7 swept-tapered flying wing (30° LE sweep, 0.5 taper ratio,
n_span = 12 panels, single chordwise row so all panels shed into the wake).

The free-wake iteration replaces the fixed semi-infinite trailing legs with
finite vortex segments that are advected by the local induced velocity until
CDi stabilises.  One chord-length row spacing (ds = c_ref) gives clearly
visible wake deformation per iteration.

Outputs (saved alongside this script)
--------------------------------------
05_wake_shape.png      — top-view wake: streamlines at α = 12° (fixed vs relaxed)
05_wake_convergence.png— CDi per relaxation iteration at α = 12°
05_CDi_comparison.png  — CDi(α) sweep: fixed vs relaxed wake

Run from the repo root::

    python examples/05_wake_relaxation/05_wake_relaxation.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lspaero.geometry.wing import make_vlm_mesh
from lspaero.solver.solve import solve_vlm, solve_vlm_relaxed
from lspaero.solver.wake import trace_streamlines

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Wing geometry (AR ≈ 6.7 conventional, shown as 13.3 with half-wing S_ref) ─
HALF_SPAN  = 5.0
ROOT_CHORD = 2.0
TIP_CHORD  = 1.0   # taper ratio 0.5
SWEEP_LE   = 30.0
TWIST_TIP  = -2.0
N_SPAN     = 12    # single chordwise row → all panels shed into the wake
N_CHORD    = 1

mesh = make_vlm_mesh(
    half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
    sweep_le=SWEEP_LE, twist_tip=TWIST_TIP,
    n_span=N_SPAN, n_chord=N_CHORD,
)

S_ref = float(mesh.areas.sum())
b_ref = 2.0 * float(mesh.vertices[:, 1].max())
c_ref = S_ref / (0.5 * b_ref)
x_ref = 0.25 * c_ref
AR_code = b_ref**2 / S_ref   # = 2 × conventional AR (half-wing S_ref convention)

print(f"Wing: AR(code) = {AR_code:.2f}, S_ref = {S_ref:.3f} m², "
      f"b_ref = {b_ref:.3f} m, c_ref = {c_ref:.3f} m")
print(f"  (conventional AR = {AR_code/2:.1f}, Λ = {SWEEP_LE}°, "
      f"taper = {TIP_CHORD/ROOT_CHORD:.2f})")

# ── Wake relaxation parameters ────────────────────────────────────────────────
# ds = c_ref gives one chord-length per wake row → visible deformation per iter.
ALPHA_DEMO = 12.0   # high-α case; CDi increases clearly with wake roll-up
N_WAKE     = 20
DS         = c_ref          # one chord per wake row
N_ITER     = 40
RELAX      = 0.3
N_FIXED    = 2
N_MIN_ITER = 4      # minimum iters before convergence check (lets wake build up)
TOL        = 1e-5   # CDi fractional change convergence criterion

# ── Solve fixed and relaxed at ALPHA_DEMO ─────────────────────────────────────
print(f"\n── α = {ALPHA_DEMO}°: fixed-wake vs relaxed-wake ──")
r_fixed = solve_vlm(
    mesh, alpha_deg=ALPHA_DEMO,
    S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref,
)

r_relax = solve_vlm_relaxed(
    mesh, alpha_deg=ALPHA_DEMO,
    S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref,
    n_wake_rows=N_WAKE, ds=DS,
    n_iter=N_ITER, relax=RELAX, n_fixed_rows=N_FIXED,
    n_min_iter=N_MIN_ITER, tol=TOL,
)

delta_CDi = (r_relax["CDi"] - r_fixed["CDi"]) / r_fixed["CDi"] * 100
print(f"  Fixed  wake:  CL = {r_fixed['CL']:.4f}, CDi = {r_fixed['CDi']:.5f}")
print(f"  Relaxed wake: CL = {r_relax['CL']:.4f}, CDi = {r_relax['CDi']:.5f}  "
      f"({r_relax['n_iters']} iters)")
print(f"  ΔCDi = {delta_CDi:+.2f}%")

# ── Plot 1: Wake shape (top view, x-y plane) ─────────────────────────────────
A_fix    = r_fixed["A"]
B_fix    = r_fixed["B"]
Gam_fix  = r_fixed["Gamma"]
A_rel    = r_relax["A"]
B_rel    = r_relax["B"]
Gam_rel  = r_relax["Gamma"]
V_inf    = r_fixed["V_inf"]
V_inf_r  = r_relax["V_inf"]

n_trace = N_WAKE

# Fixed wake: streamlines traced in fixed-wake velocity field
lines_A_fix, lines_B_fix = trace_streamlines(
    A_fix, B_fix, Gam_fix, V_inf, n_trace, DS,
)
# Relaxed wake: streamlines traced in relaxed-wake velocity field
lines_A_rel, lines_B_rel = trace_streamlines(
    A_rel, B_rel, Gam_rel, V_inf_r, n_trace, DS,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, label, lines_A, lines_B, res in [
    (axes[0], "Fixed wake",   lines_A_fix, lines_B_fix, r_fixed),
    (axes[1], "Relaxed wake", lines_A_rel, lines_B_rel, r_relax),
]:
    # Wake streamlines (left = A side, right = B side)
    for k in range(N_SPAN):
        ax.plot(lines_A[:, k, 0], lines_A[:, k, 1], "b-", lw=0.9, alpha=0.5)
        ax.plot(lines_B[:, k, 0], lines_B[:, k, 1], "r-", lw=0.9, alpha=0.5)

    # Wing 1/4-chord bound segments
    for k in range(N_SPAN):
        ax.plot([A_fix[k, 0], B_fix[k, 0]],
                [A_fix[k, 1], B_fix[k, 1]], "k-", lw=1.5)

    ax.set_xlabel("x  (chordwise, m)")
    ax.set_ylabel("y  (spanwise, m)")
    ax.set_title(
        f"{label}  —  α = {ALPHA_DEMO}°\n"
        f"CL = {res['CL']:.4f},  CDi = {res['CDi']:.5f}"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Wake streamlines — Λ = {SWEEP_LE}°, taper = {TIP_CHORD/ROOT_CHORD:.1f}, "
    f"n_wake_rows = {N_WAKE}",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "05_wake_shape.png"), dpi=150)
plt.close(fig)
print("\nSaved 05_wake_shape.png")

# ── Plot 2: CDi per iteration (convergence history) ───────────────────────────
history = r_relax["history"]   # CDi per iteration

fig, ax = plt.subplots(figsize=(7, 4.5))
iters = np.arange(1, len(history) + 1)
ax.plot(iters, history, "b-o", ms=5, lw=1.8, label="CDi per iteration")
ax.axhline(r_fixed["CDi"], color="k", lw=1.2, ls="--",
           label=f"Fixed wake  CDi = {r_fixed['CDi']:.5f}")
ax.axhline(r_relax["CDi"], color="r", lw=1.0, ls=":",
           label=f"Converged   CDi = {r_relax['CDi']:.5f}")
ax.set_xlabel("Relaxation iteration")
ax.set_ylabel(r"$C_{Di}$")
ax.set_title(
    f"Wake-relaxation convergence  —  α = {ALPHA_DEMO}°\n"
    f"n_wake_rows = {N_WAKE},  ds = c_ref,  relax = {RELAX}"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "05_wake_convergence.png"), dpi=150)
plt.close(fig)
print("Saved 05_wake_convergence.png")

# ── Plot 3: CDi(α) sweep — fixed vs relaxed ───────────────────────────────────
alphas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
CDi_fixed   = []
CDi_relaxed = []
CL_fixed    = []

print(f"\n{'α':>5}  {'CL':>8}  {'CDi_fix':>9}  {'CDi_rel':>9}  {'ΔCDi%':>8}")
for a in alphas:
    rf = solve_vlm(
        mesh, alpha_deg=a,
        S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref,
    )
    rr = solve_vlm_relaxed(
        mesh, alpha_deg=a,
        S_ref=S_ref, b_ref=b_ref, c_ref=c_ref, x_ref=x_ref,
        n_wake_rows=N_WAKE, ds=DS,
        n_iter=N_ITER, relax=RELAX, n_fixed_rows=N_FIXED,
        n_min_iter=N_MIN_ITER, tol=TOL,
    )
    CDi_fixed.append(rf["CDi"])
    CDi_relaxed.append(rr["CDi"])
    CL_fixed.append(rf["CL"])
    dCDi = (rr["CDi"] - rf["CDi"]) / max(rf["CDi"], 1e-10) * 100
    print(f"{a:5.1f}  {rf['CL']:8.4f}  {rf['CDi']:9.5f}  {rr['CDi']:9.5f}  {dCDi:+8.2f}%")

CL_arr  = np.array(CL_fixed)
CDi_ell = CL_arr**2 / (np.pi * AR_code)   # elliptic reference using AR_code convention

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(alphas, CDi_fixed,    "b-o", ms=5, lw=1.8, label="Fixed wake")
ax.plot(alphas, CDi_relaxed,  "r-s", ms=5, lw=1.8, label="Relaxed wake")
ax.plot(alphas, CDi_ell,      "k--", lw=1.2,
        label=r"Elliptic ref  $C_L^2 / (\pi AR)$")
ax.set_xlabel("α  (degrees)")
ax.set_ylabel(r"$C_{Di}$")
ax.set_title(
    f"Induced drag — fixed vs relaxed wake\n"
    f"AR = {AR_code:.1f}, Λ = {SWEEP_LE}°, n_wake = {N_WAKE} rows, ds = c_ref"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "05_CDi_comparison.png"), dpi=150)
plt.close(fig)
print("\nSaved 05_CDi_comparison.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Stage 6 summary ===")
print(f"Wing: AR(code) = {AR_code:.2f} (conv. AR = {AR_code/2:.1f}), "
      f"Λ = {SWEEP_LE}°, taper = {TIP_CHORD/ROOT_CHORD:.2f}")
print(f"Fixed  wake:  CL = {r_fixed['CL']:.4f},  CDi = {r_fixed['CDi']:.5f}")
print(f"Relaxed wake: CL = {r_relax['CL']:.4f},  CDi = {r_relax['CDi']:.5f}  "
      f"({r_relax['n_iters']} iters)")
print(f"ΔCDi (α = {ALPHA_DEMO}°) = {delta_CDi:+.2f}%")
