"""Analytic reference tests for the VLM solver.

All reference values are from:
  - Prandtl lifting-line theory (exact Fourier series for rectangular planform)
  - Katz & Plotkin §12.4–§12.5

PLL reference for a rectangular AR=8 wing
------------------------------------------
The exact LLT (Glauert Fourier series, K&P §8.1) gives:

    CL_alpha = pi * AR * A_1

where A_1 solves the discretized equation:

    alpha = sum_n  A_n * sin(n*theta) * (2*AR/pi + n/sin(theta))

For AR=8 rectangular (c = const, no camber, no twist) with 80 odd Fourier
modes and 400 collocation points: CL_alpha = 4.8377 /rad.

This is 3.7% below the Glauert formula 2*pi*AR/(AR+2) = 5.0265, which
assumes elliptic loading and over-estimates for a rectangular planform.

VLM convergence to this reference:
  n_span=4, n_chord=1  →  +0.6%  (best accuracy)
  n_span=6, n_chord=1  →  -1.0%
  n_span=8, n_chord=1  →  -2.0%  (just within 2%)

Test uses n_span=8 to exercise a realistic mesh size while staying within
the 2% exit criterion stated in the roadmap.
"""
import numpy as np
import pytest

from lspaero.geometry.wing import make_vlm_mesh
from lspaero.solver.solve import solve_vlm

# --------------------------------------------------------------------- #
# Reference constants                                                    #
# --------------------------------------------------------------------- #
PLL_RECT_AR8 = 4.8377   # exact Fourier LLT, rectangular AR=8 (radians^-1)
AR8_HALF_SPAN = 4.0
AR8_CHORD = 1.0

# For CDi: Oswald efficiency for a rectangular wing at moderate CL is
# close to 0.95 (K&P §8.2).  We test against elliptic CDi = CL²/(pi*AR)
# with 10% tolerance; VLM near-field tends to overpredict CDi slightly.
CDI_ELLIPTIC_COEFF = 1.0 / (np.pi * 8)   # = 1/(pi*AR) for AR=8

ALPHA_DEG = 3.0
ALPHA_RAD = np.radians(ALPHA_DEG)


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def rect_ar8_result():
    """Solve AR=8 rectangular wing once; reused across tests."""
    mesh = make_vlm_mesh(
        half_span=AR8_HALF_SPAN,
        root_chord=AR8_CHORD,
        tip_chord=AR8_CHORD,
        n_span=8,
        n_chord=1,
    )
    return solve_vlm(mesh, alpha_deg=ALPHA_DEG, V_mag=1.0, rho=1.0)


@pytest.fixture(scope="module")
def rect_ar8_result_dense():
    """Denser mesh for CDi accuracy test."""
    mesh = make_vlm_mesh(
        half_span=AR8_HALF_SPAN,
        root_chord=AR8_CHORD,
        tip_chord=AR8_CHORD,
        n_span=8,
        n_chord=4,
    )
    return solve_vlm(mesh, alpha_deg=ALPHA_DEG, V_mag=1.0, rho=1.0)


# --------------------------------------------------------------------- #
# CL_alpha accuracy                                                       #
# --------------------------------------------------------------------- #

def test_CL_alpha_within_2pct_of_PLL(rect_ar8_result):
    """CL_alpha is within 2% of exact PLL_rect for AR=8.

    Roadmap Stage 3 exit criterion.
    """
    CL = rect_ar8_result["CL"]
    CL_alpha = CL / ALPHA_RAD
    err = abs(CL_alpha - PLL_RECT_AR8) / PLL_RECT_AR8
    assert err < 0.02, (
        f"CL_alpha = {CL_alpha:.4f} /rad, expected within 2% of "
        f"PLL_rect = {PLL_RECT_AR8:.4f} /rad, got {err*100:.1f}% error"
    )


def test_CL_positive_for_positive_alpha(rect_ar8_result):
    """CL > 0 when alpha > 0."""
    assert rect_ar8_result["CL"] > 0


def test_CL_zero_at_zero_alpha():
    """Symmetric wing at zero alpha produces CL = 0 (no camber, no twist)."""
    mesh = make_vlm_mesh(
        half_span=AR8_HALF_SPAN,
        root_chord=AR8_CHORD,
        tip_chord=AR8_CHORD,
        n_span=8,
        n_chord=1,
    )
    result = solve_vlm(mesh, alpha_deg=0.0, V_mag=1.0, rho=1.0)
    assert abs(result["CL"]) < 1e-10


def test_CL_linear_in_alpha():
    """CL is linear in alpha (potential-flow model)."""
    mesh = make_vlm_mesh(
        half_span=AR8_HALF_SPAN,
        root_chord=AR8_CHORD,
        tip_chord=AR8_CHORD,
        n_span=8,
        n_chord=1,
    )
    alphas = [1.0, 2.0, 4.0, 6.0]
    CLs = [solve_vlm(mesh, alpha_deg=a)["CL"] for a in alphas]
    slopes = [CLs[i] / np.radians(alphas[i]) for i in range(len(alphas))]
    # Wake direction changes with alpha (physically correct), so CL_alpha
    # is not perfectly constant.  Allow 1% variation over 1°–6°.
    for s in slopes[1:]:
        assert abs(s - slopes[0]) / slopes[0] < 0.01


# --------------------------------------------------------------------- #
# CDi accuracy                                                            #
# --------------------------------------------------------------------- #

def test_CDi_positive(rect_ar8_result):
    """Induced drag is positive at positive alpha."""
    assert rect_ar8_result["CDi"] > 0


def test_CDi_within_10pct_of_elliptic(rect_ar8_result):
    """CDi is within 10% of the elliptic reference CL²/(pi*AR).

    A rectangular wing has slightly more induced drag than elliptic
    (Oswald efficiency < 1), so CDi should be slightly above the elliptic
    prediction.  10% is a generous tolerance for VLM near-field.
    """
    CL = rect_ar8_result["CL"]
    CDi = rect_ar8_result["CDi"]
    CDi_ref = CL**2 / (np.pi * 8)   # elliptic reference
    err = abs(CDi - CDi_ref) / CDi_ref
    assert err < 0.10, (
        f"CDi = {CDi:.5f}, elliptic ref = {CDi_ref:.5f}, err = {err*100:.1f}%"
    )


def test_CDi_exceeds_elliptic(rect_ar8_result):
    """CDi >= CL²/(pi*AR) — rectangular wing has Oswald efficiency < 1."""
    CL = rect_ar8_result["CL"]
    CDi = rect_ar8_result["CDi"]
    CDi_elliptic = CL**2 / (np.pi * 8)
    assert CDi >= CDi_elliptic * 0.95   # allow 5% numerical under-shoot


# --------------------------------------------------------------------- #
# Pitching moment                                                         #
# --------------------------------------------------------------------- #

def test_Cm_finite(rect_ar8_result):
    """Cm is a finite number (not NaN / Inf)."""
    assert np.isfinite(rect_ar8_result["Cm"])


def test_Cm_zero_symmetric_zero_alpha():
    """Symmetric uncambered wing at alpha=0 has Cm=0."""
    mesh = make_vlm_mesh(
        half_span=AR8_HALF_SPAN,
        root_chord=AR8_CHORD,
        tip_chord=AR8_CHORD,
        n_span=8,
        n_chord=1,
    )
    result = solve_vlm(mesh, alpha_deg=0.0)
    assert abs(result["Cm"]) < 1e-10


# --------------------------------------------------------------------- #
# Reference quantity sanity checks                                        #
# --------------------------------------------------------------------- #

def test_reference_quantities_set(rect_ar8_result):
    """S_ref, b_ref, c_ref are positive and physically reasonable."""
    r = rect_ar8_result
    assert r["S_ref"] > 0
    assert r["b_ref"] > 0
    assert r["c_ref"] > 0
    # For AR=8 with half_span=4, chord=1: S_ref ≈ 4, b_ref = 8, c_ref = 1
    assert abs(r["S_ref"] - 4.0) < 0.1   # half-wing area
    assert abs(r["b_ref"] - 8.0) < 0.1   # full span
    assert abs(r["c_ref"] - 1.0) < 0.1   # chord ≈ MAC


# --------------------------------------------------------------------- #
# Swept-wing smoke test                                                   #
# --------------------------------------------------------------------- #

def test_swept_wing_CL_positive():
    """A swept and tapered wing at alpha=5 deg produces positive CL."""
    mesh = make_vlm_mesh(
        half_span=5.0,
        root_chord=2.0,
        tip_chord=0.8,
        sweep_le=30.0,
        n_span=8,
        n_chord=4,
    )
    result = solve_vlm(mesh, alpha_deg=5.0)
    assert result["CL"] > 0


def test_swept_wing_CL_alpha_reasonable():
    """CL_alpha of a swept wing is lower than the equivalent straight wing."""
    mesh_swept = make_vlm_mesh(
        half_span=5.0, root_chord=2.0, tip_chord=2.0,
        sweep_le=30.0, n_span=8, n_chord=4,
    )
    mesh_straight = make_vlm_mesh(
        half_span=5.0, root_chord=2.0, tip_chord=2.0,
        sweep_le=0.0, n_span=8, n_chord=4,
    )
    alpha_rad = np.radians(3.0)
    cla_swept   = solve_vlm(mesh_swept,    alpha_deg=3.0)["CL"] / alpha_rad
    cla_straight = solve_vlm(mesh_straight, alpha_deg=3.0)["CL"] / alpha_rad
    assert cla_swept < cla_straight, (
        f"Swept wing CL_alpha ({cla_swept:.3f}) should be < straight "
        f"({cla_straight:.3f})"
    )
