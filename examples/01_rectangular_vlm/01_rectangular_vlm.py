"""Stage 3 demo: VLM on a rectangular wing, AR = 8.

Validates the VLM solver against Prandtl lifting-line theory (exact Fourier
series result for a rectangular planform, K&P §8.1).

Reference values
----------------
PLL_rect (AR=8, exact Fourier):   CL_alpha = 4.8377 /rad
Glauert formula (elliptic approx): CL_alpha = 5.0265 /rad  [upper bound]

VLM converges to within 2% of PLL_rect at n_span=6–8.  Accuracy is best
around n_span=4–6 (cosine spacing concentrates panels; adding more panels
eventually reduces accuracy due to the tip singularity).

Outputs (saved in same directory as this script)
-------------------------------------------------
01_CL_vs_alpha.png   — CL vs alpha, VLM vs PLL reference
01_spanwise_CL.png   — Spanwise strip-CL distribution at alpha=6°
01_convergence.png   — CL_alpha vs n_span panel count
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
# Wing parameters                                                              #
# --------------------------------------------------------------------------- #
HALF_SPAN  = 4.0   # m
CHORD      = 1.0   # m (rectangular → tip chord == root chord)
AR         = (2 * HALF_SPAN) / CHORD   # = 8

PLL_RECT   = 4.8377   # exact LLT Fourier, K&P §8.1 (rad^-1)
GLAUERT    = 2 * np.pi * AR / (AR + 2)  # 5.0265 (elliptic approximation)

# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #

def _pll_CL(alpha_deg):
    return PLL_RECT * np.radians(alpha_deg)


# --------------------------------------------------------------------------- #
# 1.  CL vs alpha                                                              #
# --------------------------------------------------------------------------- #
alphas = np.linspace(-8, 16, 25)
mesh = make_vlm_mesh(half_span=HALF_SPAN, root_chord=CHORD, tip_chord=CHORD,
                     n_span=8, n_chord=1)
CLs  = [solve_vlm(mesh, alpha_deg=a)["CL"] for a in alphas]
CDis = [solve_vlm(mesh, alpha_deg=a)["CDi"] for a in alphas]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(alphas, CLs, "b-o", ms=4, label="VLM (n=8×1)")
ax.plot(alphas, _pll_CL(alphas), "k--", lw=1.5,
        label=f"PLL (exact, {PLL_RECT:.3f} /rad)")
ax.plot(alphas, GLAUERT * np.radians(alphas), "r:", lw=1,
        label=f"Glauert elliptic ({GLAUERT:.3f} /rad)")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("α  (deg)")
ax.set_ylabel("CL")
ax.set_title(f"Rectangular wing  AR = {AR:.0f}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
CL_arr = np.array(CLs)
CDi_arr = np.array(CDis)
cdi_ref = CL_arr**2 / (np.pi * AR)
ax.plot(CL_arr, CDi_arr, "b-o", ms=4, label="VLM (near-field K-J)")
ax.plot(CL_arr[CL_arr >= 0], cdi_ref[CL_arr >= 0], "k--", lw=1.5,
        label="Elliptic: CL²/(πAR)")
ax.set_xlabel("CL")
ax.set_ylabel("CDi")
ax.set_title("Drag polar")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "01_CL_vs_alpha.png"), dpi=150)
plt.close(fig)
print(f"Saved 01_CL_vs_alpha.png")


# --------------------------------------------------------------------------- #
# 2.  Spanwise loading at alpha = 6°                                           #
# --------------------------------------------------------------------------- #
alpha_plot = 6.0
mesh8 = make_vlm_mesh(half_span=HALF_SPAN, root_chord=CHORD, tip_chord=CHORD,
                      n_span=16, n_chord=1)
res8 = solve_vlm(mesh8, alpha_deg=alpha_plot)

dCL = res8["dCL"]          # (Np,)  panel lift contribution / (q * S_ref)
A   = res8["A"]
B   = res8["B"]

# Spanwise coordinate at bound-vortex midpoint
y_mid = 0.5 * (A[:, 1] + B[:, 1])   # (Np,)
dy    = B[:, 1] - A[:, 1]            # (Np,) panel spanwise width
b2    = HALF_SPAN
eta   = y_mid / b2
deta  = dy / b2                       # normalised panel width in η

# Spanwise loading = dCL / dη.  dCL ∝ Γ·Δy so dividing by Δη gives the
# loading distribution ∝ Γ(y), which peaks at the root for a symmetric wing.
CL8          = res8["CL"]
loading      = dCL / deta             # (Np,)  lift per unit η

# Elliptic reference: ∝ sqrt(1 - η²), normalised to match VLM at root
eta_ref      = np.linspace(0, 1, 200)
elliptic_ref = np.sqrt(np.maximum(1 - eta_ref**2, 0))
scale        = loading[0] / elliptic_ref[0]   # match at root (η≈0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(eta, loading, "b-o", ms=4, label=f"VLM, α={alpha_plot}°, CL={CL8:.3f}")
ax.plot(eta_ref, elliptic_ref * scale, "k--", lw=1.5, label="Elliptic ref")

ax.set_xlabel("η = y / (b/2)")
ax.set_ylabel("Spanwise loading  dCL/dη")
ax.set_title(f"Spanwise loading  –  AR={AR:.0f} rectangular, α={alpha_plot}°")
ax.set_xlim(0, 1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "01_spanwise_CL.png"), dpi=150)
plt.close(fig)
print(f"Saved 01_spanwise_CL.png")


# --------------------------------------------------------------------------- #
# 3.  Convergence: CL_alpha vs n_span                                          #
# --------------------------------------------------------------------------- #
n_list = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
alpha_conv = 3.0
alpha_rad  = np.radians(alpha_conv)
cla_list = []
for n in n_list:
    m = make_vlm_mesh(half_span=HALF_SPAN, root_chord=CHORD, tip_chord=CHORD,
                      n_span=n, n_chord=1)
    r = solve_vlm(m, alpha_deg=alpha_conv)
    cla_list.append(r["CL"] / alpha_rad)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(n_list, cla_list, "b-o", ms=5, label="VLM (n_chord=1)")
ax.axhline(PLL_RECT, color="k", ls="--", lw=1.5,
           label=f"PLL_rect = {PLL_RECT:.4f} /rad")
ax.axhline(GLAUERT,  color="r", ls=":",  lw=1.2,
           label=f"Glauert  = {GLAUERT:.4f} /rad (elliptic)")
ax.fill_between(
    [min(n_list), max(n_list)],
    PLL_RECT * 0.98, PLL_RECT * 1.02,
    alpha=0.12, color="k", label="±2% band"
)
ax.set_xlabel("n_span  (spanwise panels)")
ax.set_ylabel("CL_α  (rad⁻¹)")
ax.set_title(f"VLM convergence — rectangular AR={AR:.0f}")
ax.set_ylim(4.3, 5.3)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "01_convergence.png"), dpi=150)
plt.close(fig)
print(f"Saved 01_convergence.png")


# --------------------------------------------------------------------------- #
# Summary printout                                                              #
# --------------------------------------------------------------------------- #
mesh_ref = make_vlm_mesh(half_span=HALF_SPAN, root_chord=CHORD, tip_chord=CHORD,
                         n_span=8, n_chord=1)
ref      = solve_vlm(mesh_ref, alpha_deg=5.0)
cla_vlm  = ref["CL"] / np.radians(5.0)

print()
print("=" * 55)
print(f"  Rectangular wing  AR = {AR:.0f}")
print(f"  n_span=8, n_chord=1")
print("=" * 55)
print(f"  PLL_rect  (exact Fourier) : {PLL_RECT:.4f} /rad")
print(f"  Glauert formula (elliptic): {GLAUERT:.4f} /rad")
print(f"  VLM CL_alpha              : {cla_vlm:.4f} /rad")
print(f"  Error vs PLL_rect         : {100*(cla_vlm-PLL_RECT)/PLL_RECT:+.1f}%")
print(f"  CL  at alpha=5°           : {ref['CL']:.4f}")
print(f"  CDi at alpha=5°           : {ref['CDi']:.5f}")
print(f"  Cm  at alpha=5°           : {ref['Cm']:.4f}")
print("=" * 55)
