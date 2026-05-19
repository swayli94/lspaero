"""Vectorized Biot-Savart law for straight vortex filaments.

All formulae from Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed. (2001):
  - Finite segment: Eq. 2.69
  - Semi-infinite segment: §2.14 (limit of Eq. 2.69 as one end → ∞)

Broadcasting convention
-----------------------
P, A, B can have arbitrary matching leading dimensions (..., 3).
The returned velocity has the same shape as the broadcast of the inputs.
"""
from __future__ import annotations

import numpy as np

_FOUR_PI = 4.0 * np.pi
# Vortex-core desingularisation: |r₁×r₂|² is clamped to this minimum.
# Equivalent to a Rankine vortex core of radius ≈ sqrt(_CORE_SQ).
# Set to (1e-5)^2 to handle near-field and self-induction cleanly.
_CORE_SQ = 1e-10


def vel_segment(
    P: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Induced velocity of a unit-strength (Γ=1) finite vortex segment A→B at P.

    Katz & Plotkin Eq. (2.69):

        V = (1/4π) · (r₁×r₂) / |r₁×r₂|² · r₀·(r₁/|r₁| − r₂/|r₂|)

    where r₀ = B−A, r₁ = P−A, r₂ = P−B.

    Parameters
    ----------
    P : (..., 3) evaluation points
    A : (..., 3) segment start (broadcasts with P and B)
    B : (..., 3) segment end

    Returns
    -------
    vel : (..., 3)
        Induced velocity for unit circulation.  Multiply by Γ for physical vel.
    """
    r1 = P - A                                              # (..., 3)
    r2 = P - B

    cross = np.cross(r1, r2)                               # (..., 3)
    cross_sq = np.einsum("...i,...i->...", cross, cross)   # (...,)
    cross_sq = np.maximum(cross_sq, _CORE_SQ)

    r1_mag = np.sqrt(np.einsum("...i,...i->...", r1, r1))  # (...,)
    r2_mag = np.sqrt(np.einsum("...i,...i->...", r2, r2))
    r1_mag = np.maximum(r1_mag, 1e-12)
    r2_mag = np.maximum(r2_mag, 1e-12)

    r0 = B - A                                             # (..., 3)
    dot1 = np.einsum("...i,...i->...", r0, r1) / r1_mag   # r₀·r₁/|r₁|
    dot2 = np.einsum("...i,...i->...", r0, r2) / r2_mag   # r₀·r₂/|r₂|

    factor = (dot1 - dot2) / (_FOUR_PI * cross_sq)        # (...,)
    return factor[..., None] * cross                       # (..., 3)


def vel_semi_inf(
    P: np.ndarray,
    A: np.ndarray,
    d_hat: np.ndarray,
) -> np.ndarray:
    """Induced velocity of a unit-strength semi-infinite vortex filament A→∞.

    The filament starts at A and extends to +∞ in the direction d̂.

    Katz & Plotkin §2.14:

        V = (1/4π) · (r₁×d̂) / |r₁×d̂|² · (1 + r̂₁·d̂)

    where r₁ = P−A.

    Parameters
    ----------
    P : (..., 3)
    A : (..., 3)
    d_hat : (..., 3) or (3,)  wake direction (normalised internally)

    Returns
    -------
    vel : (..., 3)
    """
    d = d_hat / np.linalg.norm(d_hat, axis=-1, keepdims=True)
    r1 = P - A                                              # (..., 3)

    # Limit of finite-segment formula as B→∞: r₁×r₂ → -L(r₁×d), but factor
    # picks up another -L, so the net cross is d×r₁ (not r₁×d).
    cross = np.cross(d, r1)                                # (..., 3)  d × r₁
    cross_sq = np.einsum("...i,...i->...", cross, cross)   # (...,)
    cross_sq = np.maximum(cross_sq, _CORE_SQ)

    r1_mag = np.sqrt(np.einsum("...i,...i->...", r1, r1))  # (...,)
    r1_mag = np.maximum(r1_mag, 1e-12)

    cos_t = np.einsum("...i,...i->...", r1 / r1_mag[..., None], d)  # r̂₁·d̂

    factor = (1.0 + cos_t) / (_FOUR_PI * cross_sq)
    return factor[..., None] * cross                       # (..., 3)
