"""Analytic reference tests for biot_savart.py.

Reference formulae from Katz & Plotkin §2.13-§2.14.
"""
import numpy as np
import pytest

from lspaero.solver.biot_savart import vel_segment, vel_semi_inf


def test_segment_perpendicular_bisector():
    """A point on the perpendicular bisector of a unit-length segment on the z-axis.

    Segment from A=(0,0,-0.5) to B=(0,0,0.5), evaluation at P=(1,0,0).
    Analytic result (K&P Eq. 2.69 evaluated by hand):

      r1 = (1,0,-0.5), r2 = (1,0,0.5)
      r0 = (0,0,1)
      r1×r2 = (0*0.5−(−0.5)*0, (−0.5)*1−1*0.5, 1*0−0*1) = (0, −1, 0)
      |r1×r2|² = 1
      dot1 = r0·r1/|r1| = −0.5/√1.25 = −0.5/1.118..
      dot2 = r0·r2/|r2| =  0.5/√1.25
      V = (1/4π)*(0,−1,0)*1*(−1/√1.25 ) = ... let's verify numerically.

    The expected y-component magnitude equals (1/4π·r)*(cos θ₁ + cos θ₂) where
    r=1 is the perpendicular distance and cos θ = 0.5/√1.25 for both endpoints.
    """
    A = np.array([0.0, 0.0, -0.5])
    B = np.array([0.0, 0.0,  0.5])
    P = np.array([1.0, 0.0,  0.0])

    v = vel_segment(P, A, B)

    r = 1.0
    cos_t = 0.5 / np.sqrt(1.25)
    v_mag_expected = (1.0 / (4.0 * np.pi * r)) * 2.0 * cos_t

    assert abs(np.linalg.norm(v) - v_mag_expected) < 1e-10
    # Segment goes in +z; at P=(1,0,0) right-hand rule gives +y induced velocity
    assert v[1] > 0
    assert abs(v[0]) < 1e-14
    assert abs(v[2]) < 1e-14


def test_segment_symmetry():
    """Reversing the segment reverses the induced velocity."""
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    P = np.array([0.5, 1.0, 0.0])

    v_fwd = vel_segment(P, A, B)
    v_rev = vel_segment(P, B, A)
    np.testing.assert_allclose(v_fwd, -v_rev, atol=1e-14)


def test_segment_superposition():
    """Two collinear half-segments sum to one full segment."""
    A = np.array([0.0, 0.0, 0.0])
    M = np.array([0.5, 0.0, 0.0])   # midpoint
    B = np.array([1.0, 0.0, 0.0])
    P = np.array([0.5, 1.0, 0.0])

    v_full = vel_segment(P, A, B)
    v_half = vel_segment(P, A, M) + vel_segment(P, M, B)
    np.testing.assert_allclose(v_full, v_half, rtol=1e-10)


def test_semi_inf_vs_long_segment():
    """A very long finite segment approximates a semi-infinite one."""
    A = np.array([0.0, 0.0, 0.0])
    B_far = np.array([1e6, 0.0, 0.0])
    P = np.array([0.0, 1.0, 0.0])
    d = np.array([1.0, 0.0, 0.0])

    v_semi = vel_semi_inf(P, A, d)
    # Semi-infinite vortex from A=(0,0,0) in +x, at P=(0,1,0):
    # Right-hand rule (thumb in +x, fingers curl): at P which is in +y, induced
    # velocity is in +z direction.  Magnitude = 1/(4π·r)·(cos 0° + cos 90°) = 1/4π.
    expected = np.array([0.0, 0.0, +1.0 / (4.0 * np.pi)])
    np.testing.assert_allclose(v_semi, expected, rtol=1e-6)

    # Long finite segment ≈ semi-infinite
    v_long = vel_segment(P, A, B_far)
    np.testing.assert_allclose(v_long, v_semi, rtol=1e-4)


def test_semi_inf_direction_dependence():
    """Semi-infinite vel is zero when P lies on the filament axis."""
    A = np.array([0.0, 0.0, 0.0])
    P = np.array([2.0, 0.0, 0.0])   # collinear with A and d
    d = np.array([1.0, 0.0, 0.0])
    v = vel_semi_inf(P, A, d)
    # r1×d = 0 → desingularised to zero velocity
    assert np.linalg.norm(v) < 1e-3   # core regularisation gives tiny value


def test_batch_shape():
    """Batch evaluation returns correct shape."""
    Ni, Nj = 7, 5
    P = np.random.rand(Ni, 1, 3)
    A = np.random.rand(1, Nj, 3)
    B = np.random.rand(1, Nj, 3)
    v = vel_segment(P, A, B)
    assert v.shape == (Ni, Nj, 3)
