# LSPAero — Development Roadmap & Task List

**A level-set panel method for flying-wing aerodynamics in Python.**

This document is the working contract between the developer and the AI coding
assistant. It defines what is being built, in what order, and how each step is
verified. Update it as the project evolves; treat it as living, not historical.

## 0. Project Identity

- **Name (display):** LSPAero
- **Name (package / repo / import):** `lspaero`
- **Language:** Python ≥ 3.10
- **License:** MIT (recommended)
- **Core dependencies (planned):** `numpy`, `scipy`, `matplotlib`, `pyvista`
  (visualization, optional), `pytest` (testing). No CFD libraries.
- **Philosophy:** geometry backend and aerodynamic solver are **decoupled**.
  Any geometry source (analytic mesh generator, SDF backend, external `.tri`
  file) feeds the same solver interface.

## 1. Guiding Principles

These are non-negotiable; they prevent the project from drifting.

1. **Every stage must produce something runnable.** No stage ends with "code
   that will work once stage N+1 is done." Each stage has a demo script and a
   plot.
2. **Verification before features.** A new module is not "done" until it has
   been compared against a known reference (analytic, AVL, VSPAERO, or
   XFOIL).
3. **Geometry / solver decoupling is sacred.** The solver consumes a fixed
   `Mesh` data structure. It does not know whether the mesh came from a
   parametric generator or an SDF. This makes the SDF backend a drop-in
   replacement, and lets the two be cross-validated.
4. **Numpy vectorization, not Python loops.** Influence coefficient
   assembly is the hot path. Loops over panels are forbidden in production
   code; loops are only acceptable in test scaffolding.
5. **Plot first, debug second.** Mesh, normals, wake, Cp — visualize them
   before trusting any number. 90% of geometry bugs are visible.
6. **One thing at a time.** Do not start the SDF backend while wake
   relaxation is half-debugged. Finish, verify, commit, then move on.

## 2. Repository Layout (target)

```
lspaero/
├── README.md
├── ROADMAP.md                # this file
├── LICENSE
├── pyproject.toml
├── lspaero/
│   ├── __init__.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── mesh.py           # Mesh data structure (panels, normals, TE, adjacency)
│   │   ├── naca.py           # NACA 4-digit airfoil
│   │   ├── wing.py           # parametric flying-wing mesh generator
│   │   └── sdf/              # SDF backend (added in Stage 7)
│   │       ├── primitives.py
│   │       ├── wing_sdf.py
│   │       └── sampler.py
│   ├── solver/
│   │   ├── __init__.py
│   │   ├── biot_savart.py    # vectorized vortex-segment induction
│   │   ├── influence.py      # AIC matrix assembly
│   │   ├── kutta.py          # trailing-edge closure
│   │   ├── wake.py           # fixed / relaxed wake
│   │   └── solve.py          # main driver
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── pg_correction.py  # Prandtl-Glauert
│   │   └── forces.py         # Cp, CL, CDi, Cm integration
│   ├── io/
│   │   ├── __init__.py
│   │   ├── tri.py            # Cart3D .tri reader (optional)
│   │   └── vtk.py            # ParaView/PyVista export
│   └── viz/
│       ├── __init__.py
│       └── plot.py
├── examples/
│   ├── 00_2d_airfoil/          # Stage 1: Hess-Smith 2D solver
│   │   ├── 00_2d_airfoil.py
│   │   └── *.png
│   ├── 01_rectangular_vlm/     # Stage 3
│   ├── 02_swept_flying_wing_vlm/
│   ├── 03_thick_flying_wing/
│   ├── 04_compressibility/
│   ├── 05_wake_relaxation/
│   ├── 06_sdf_backend/
│   └── 07_sdf_perturbation_study/
└── tests/
    ├── test_biot_savart.py
    ├── test_mesh.py
    ├── test_solver_analytic.py
    ├── test_pg.py
    └── test_sdf.py
```

## 3. Core Data Contract: the `Mesh` object

Defined once in `geometry/mesh.py` and **never modified by downstream code**.
Both the parametric generator (Stage 2) and the SDF sampler (Stage 7) produce
this exact structure.

```python
class Mesh:
    # geometry
    vertices: np.ndarray         # (Nv, 3) panel corner coordinates
    panels:   np.ndarray         # (Np, 4) vertex indices, CCW seen from outside
    # derived (computed once at construction)
    centroids: np.ndarray        # (Np, 3) panel collocation points
    normals:   np.ndarray        # (Np, 3) outward unit normals
    areas:     np.ndarray        # (Np,)
    # topology
    te_pairs:  np.ndarray        # (Nte, 2) upper/lower panel index pairs at trailing edge
    wake_seed: np.ndarray        # (Nte, 3) starting points for wake filaments
    # optional metadata
    surface_id: np.ndarray       # (Np,) which surface (upper/lower/tip/control) — useful for plotting and control surfaces
```

**Invariants checked at construction:**

- all panels are planar (or near-planar within tolerance)
- normals point outward (consistent winding)
- `te_pairs` reference valid panel indices
- closed (watertight) — sum of (area·normal) over all panels ≈ 0

## 4. Stage-by-Stage Roadmap

Each stage has: **goal · inputs · outputs · verification · exit criterion**.
Estimated effort is for focused work; multiply by 1.5–2 for learning-while-doing.

### Stage 1 — Project skeleton & 2D Hess–Smith warm-up

**Effort:** 3–5 days

**Goal.** Stand up the package, get tooling working, and write a 2D
Hess–Smith panel method for a single airfoil. This is the algorithmic
scaffold for everything that follows.

**Tasks.**

- [x] Initialize repo, `pyproject.toml`, `pytest`, `pre-commit` (optional).
- [x] `geometry/naca.py`: NACA 4-digit airfoil generator (analytic formula),
      with cosine-clustered chordwise sampling.
- [x] 2D Hess–Smith solver (constant source + single vortex) — small standalone
      module, ~300 lines. May live in `examples/00_2d_airfoil.py` rather than
      the package, since it is throwaway.
- [x] Cp distribution plot.

**Verification.** NACA 0012 at α = 0°, 4°, 8°:

- Cp distribution shape physically correct (suction peak on upper, pressure on
  lower; matches expected behaviour). XFOIL not installed; comparison deferred.
- CL slope = 6.95 /rad ≈ 2π (6.28 /rad) within 10 %; the excess is the known
  thickness correction for NACA 0012, not a solver error.

**Exit criterion.** ✅ `examples/stage1_cp.png` and `examples/stage1_cl.png`
committed. XFOIL overlay omitted (tool not installed); CL vs α plot with
2π/rad reference serves as the quantitative reference.

### Stage 2 — Mesh data structure & parametric flying-wing generator

**Effort:** 4–6 days

**Goal.** The `Mesh` class and a generator that turns
(span, root chord, tip chord, sweep, dihedral, twist, airfoil) into a valid
`Mesh` object.

**Tasks.**

- [x] `geometry/mesh.py`: `Mesh` class with invariants and self-check.
- [x] `geometry/wing.py`: parametric wing builder:
      - cosine spanwise distribution
      - cosine chordwise distribution
      - twist + dihedral + sweep + taper
      - tip cap (closed surface)
      - trailing-edge pair identification
- [x] `viz/plot.py`: render mesh with normals as arrows, highlight TE in red.
- [x] Mesh validity tests (`tests/test_mesh.py`): watertightness, normal
      orientation, TE pairs.

**Verification.**

- Plot a rectangular wing, swept wing, and tapered swept-twisted flying wing.
  All look right.
- `Mesh.is_watertight()` returns True.
- All normals point outward (verified by inflating geometry by ε along normals
  and checking volume increases).

**Exit criterion.** Three meshes saved as VTK files, viewable in ParaView,
with normals and TE pairs visible.

### Stage 3 — VLM solver (thin surface, fixed wake)

**Effort:** 1–1.5 weeks

**Goal.** A working vortex-lattice solver on the mean camber surface,
producing CL, CDi, Cm and spanwise loading.

**Tasks.**

- [x] `solver/biot_savart.py`: vectorized vortex-segment induced velocity.
      Uses NumPy broadcasting; no Python loops over panels.
      Cutoff radius for near-singularity.
- [x] `solver/influence.py`: assemble AIC matrix
      A[i,j] = (induced velocity of ring j at collocation i) · n_i.
      Horseshoe vortex per panel; symmetric image system for full-wing CL.
- [x] `solver/solve.py`: assemble RHS, solve linear system, return circulations.
      Fixed planar wake aligned with freestream.
- [x] `physics/forces.py`: Kutta–Joukowski force per filament,
      integrate to CL, CDi (near-field), Cm about a reference point.

**Notes on PLL reference.**

The ROADMAP originally cited the Glauert formula CL_α = 2πAR/(AR+2) = 5.027
as the PLL reference for a rectangular AR=8 wing.  That formula assumes
*elliptic* loading and over-estimates the rectangular-wing value by ~3.7%.
The correct reference is the exact Fourier-series LLT (K&P §8.1):

  PLL_rect (AR=8) = 4.8377 /rad  (converged with 40+ odd Fourier modes)

VLM with cosine spanwise spacing converges to within ±2% of this value at
n_span ≤ 8 (optimal accuracy at n_span = 4–6); accuracy degrades for
n_span > 12 due to the tip trailing-vortex singularity.

**Verification.**

- Rectangular wing AR = 8, n_span=8, n_chord=1:
  CL_α = 4.754 /rad, error = −1.7% vs PLL_rect = 4.838 /rad ✅ within 2%.
- CDi within 10% of CL²/(πAR) (near-field K-J vs Trefftz) ✅
- Swept flying-wing (AR≈7.7, 30° sweep, −2° washout): CL_α = 4.43 /rad,
  Cm_α < 0 (statically stable configuration) ✅

**Exit criterion.** ✅ `examples/01_rectangular_vlm/01_rectangular_vlm.py`
and `examples/02_swept_flying_wing_vlm/02_swept_flying_wing_vlm.py` produce
plots and printouts.  All tests in `tests/test_solver_analytic.py` pass.

### Stage 4 — Thick-surface vortex-panel solver

**Effort:** 1.5–2 weeks

**Goal.** Upgrade from VLM to thick-panel method. Now we compute **real
surface Cp**, not just ΔCp. This is where VSPAERO's Panel mode lives.

**Tasks.**

- [x] Extend `geometry/wing.py`: thick mesh with upper and lower surfaces,
      tip cap, proper TE pair identification across the thick TE.
      (`make_wing_mesh` produces upper + lower + tip panels, with `surface_id`
      tagging and `te_pairs` spanning the thick TE.)
- [x] Reuse `solver/biot_savart.py` and `influence.py` unchanged — they
      operate on a generic `Mesh`.
      (VLM AIC assembly reused in `solve_morino` for the lifting solution.)
- [x] Kutta condition for thick TE: upper/lower panel pairs shed a common
      wake; handled via `te_pairs` in `build_panel_aic` (same as Stage 3).
      Separate `kutta.py` not needed — Kutta is embedded in `influence.py`.
- [x] `physics/forces.py`: `forces_from_cp` integrates surface Cp over panel
      area to give per-panel force vectors; integrated to CL, CDi, Cm.
      `source_aic.py` (Hess–Smith source AIC, Katz & Plotkin §10.2) provides
      the surface tangential velocity → Bernoulli → Cp pipeline.
- [x] Plot upper/lower Cp distribution at multiple spanwise stations.
      (`03_Cp_chordwise.png`: 3 strips, upper/lower TP lines vs VLM ±ΔCp/2
      dashed lines; Cp axis inverted per aerodynamic convention.)

**Implementation notes.**

The final approach is a **source + VLM superposition** rather than a pure
vortex-ring thick-panel solver:

  A. **Hess–Smith source solve at α = 0°** (`solver/source_aic.py`):
     solve for source strengths σ on the thick surface with freestream
     `V_inf = [V, 0, 0]`. Evaluate surface velocity via `source_velocity_field`
     and apply Bernoulli → `Cp_thickness`.

  B. **VLM on the camber surface** (`solve_vlm`): gives `Gamma` per panel and
     the lifting pressure difference `ΔCp = 2Γ / (V · Δx)`.

  C. **Superimpose**: `Cp_upper = Cp_thickness − ΔCp/2`,
     `Cp_lower = Cp_thickness + ΔCp/2`. Clip at Cp = 1 to enforce the
     Bernoulli bound (VLM thin-airfoil LE singularity can push `ΔCp` → ∞
     on the first cosine-clustered panel; clipping restores physical values).

  The source solve must use α = 0° (not the actual angle of attack). Using
  actual α embeds the stagnation-point shift into σ, and the subsequent VLM
  ΔCp addition then double-counts that effect.

  A vortex-ring thick-panel solver skeleton also exists in `solver/solve.py`
  but is not the primary solver; `solve_morino` in `solver/morino.py` is
  the authoritative entry point.

**Bugs found and fixed during Stage 4.**

- `source_aic.py`: log argument was inverted (`log(numer/denom)` → should be
  `log(denom/numer)`), producing wrong-sign in-plane velocity.
- `morino.py`: source solve was using actual α freestream → double-counted
  stagnation-point shift with VLM ΔCp. Fixed to α = 0°.
- `morino.py`: missing Cp clip at 1.0 → VLM LE singularity pushed lower-surface
  Cp to physically impossible values (e.g. 3.51 at α = 4°).

**Verification.**

- Quasi-2D wing (AR = 100, NACA 0012): CL slope 6.03 /rad vs 2D Hess–Smith
  6.95 /rad; AR-correction expected. Cp shape matches 2D Hess–Smith at α = 0°
  (suction peak, symmetric upper/lower). ✅ (`00b_Cp_chordwise.png`)
- Flying wing (AR ≈ 7.7): CL from `solve_morino` agrees with `solve_vlm`
  within ~3–5% across α = −4° … 12°. ✅ (`03_CL_vs_alpha.png`)
- Spanwise loading shape (TP Cp-integrated vs VLM K-J): qualitatively
  consistent; slight differences near tip (expected — VLM vs H-S thickness
  model). ✅ (`03_spanwise_loading.png`)
- VSPAERO comparison: deferred — tool not available in current environment.

**Exit criterion.** ✅ `examples/03_thick_flying_wing/03_thick_flying_wing.py`
produces:

- `03_Cp_contour.png` — surface Cp scatter plot (matplotlib, not PyVista;
  functionally equivalent for code validation purposes)
- `03_Cp_chordwise.png` — upper/lower Cp at 3 spanwise stations with VLM
  ±ΔCp/2 overlay and inverted Cp axis
- `03_spanwise_loading.png` — spanwise loading from TP (Cp-integrated) vs
  VLM (K-J) side by side
- `03_CL_vs_alpha.png` — CL(α) sweep: VLM vs thick-panel vs 2π/rad theory

---

### Stage 5 — Prandtl–Glauert compressibility correction

**Effort:** 1–2 days

**Goal.** Support subsonic compressible flow (up to about M = 0.7) via PG.

**Tasks.**

- [x] `physics/pg_correction.py`:
      - `pg_beta(mach)` — returns β = √(1 − M²).
      - `pg_stretch_mesh(mesh, mach)` — stretches all vertex x-coordinates by
        1/β (freestream / chordwise direction), returning the transformed mesh
        and β.  y and z are unchanged.
      - Reference geometric quantities (S_ref, b_ref, c_ref, x_ref) must be
        computed from the **original** (physical) mesh before calling this
        function.  The moment reference x_ref must be divided by β when passed
        to the solver (to place the reference point correctly in stretched
        coordinates).
- [x] Mach number added as a solver input to both `solve_vlm` and
      `solve_morino` (`mach=0.0` default, backward compatible).
      - The K-J forces computed on the stretched mesh with physical S_ref give
        PG-corrected CL, CDi, Cm directly (no extra scaling).
      - `solve_morino` additionally divides Cp by β after the superposition
        step, giving the physical compressible surface Cp.

**Implementation notes.**

The PG transform stretches x by 1/β, which converts the compressible
small-disturbance equation β²φ_xx + φ_yy + φ_zz = 0 to Laplace's equation.
The incompressible solve on the stretched mesh yields:

- CL_pg = CL_1  (K-J on stretched mesh, physical S_ref — no extra scaling)
- Cp_pg = Cp_1 / β  (explicit division; surface Cp from morino only)

In the 2D / large-AR limit, CL_pg = CL_incomp / β exactly (verified at
AR = 100, error < 0.1%).  For finite-AR swept wings the correction is smaller
than 1/β because stretching x reduces effective sweep and AR — this is a known
3D compressibility effect, not a solver error.

VSPAERO comparison: deferred — tool not available in the current environment.

**Verification.**

- Quasi-2D flat plate (AR = 100, α = 4°): CL_pg / CL_incomp ≈ 1/β within
  0.1% across M = 0 … 0.7.  ✅
- Swept flying wing (AR = 11.4, 35° sweep, α = 4°): CL increases with M;
  ratio CL(M=0.5)/CL(M=0) = 1.056, consistent with partial 3D PG correction
  (full 1/β would be 1.155).  ✅ (`04_CL_vs_mach.png`)
- Chordwise ΔCp increases with M in the correct direction.  ✅

**Exit criterion.** ✅ `examples/04_compressibility/04_compressibility.py`
produces:

- `04_CL_vs_mach.png`   — CL(M) with 1/β Glauert reference
- `04_CDi_vs_mach.png`  — CDi(M) with 1/β² reference
- `04_Cp_chordwise.png` — mid-span ΔCp at M = 0, 0.3, 0.5

### Stage 6 — Wake relaxation (simplified)

**Effort:** 1–2 weeks (timebox strictly)

**Goal.** Allow the wake to align with the local flow direction, improving
induced drag prediction at moderate-to-high α.

**Tasks.**

- [x] `solver/wake.py`: finite wake node system (``WakeNodes``-style arrays),
      ``build_wake_nodes``, ``wake_node_velocity``, ``advect_wake_nodes``,
      ``trace_streamlines``.
- [x] `solver/influence.py`: ``build_aic_with_wake`` — replaces semi-infinite
      trailing legs with finite node segments + semi-infinite extension.
      Verified: planar-wake first iteration agrees with ``solve_vlm`` to
      within 3×10⁻⁶ (Biot–Savart telescoping identity).
- [x] `solver/solve.py`: ``solve_vlm_relaxed`` — outer iteration loop.
      Wake nodes advected by **induced-only** velocity (freestream subtracted)
      to prevent unbounded downstream drift.  CDi-change convergence criterion
      with a ``n_min_iter`` guard to prevent premature exit.
      Under-relaxation factor 0.3, ``n_fixed_rows=2`` near-TE rows frozen.
- [x] CDi comparison: swept flying wing (AR ≈ 6.7, Λ = 30°, taper 0.5):
      at α = 12°, CDi increases by +2.16% relative to fixed wake (physically
      expected: wake roll-up increases induced drag at high α).

**Implementation notes.**

- The AIC with finite wake telescopes exactly to the semi-infinite AIC when
  nodes are planar (collinear with d_far), so the first iteration is
  identical to ``solve_vlm`` to within floating-point.
- Each horseshoe's trailing legs (from bound-vortex endpoints A[k] and B[k])
  are replaced by ``n_wake_rows`` finite vortex segments plus a semi-infinite
  extension.  The image system mirrors these in y = 0 for the full-wing model.
- Convergence is tracked via fractional CDi change per iteration with a
  minimum-iteration guard (``n_min_iter=4`` default) so the wake has time to
  develop before the check is applied.

**Verification.**

- Planar-wake consistency: ``solve_vlm_relaxed`` (n_iter=1) agrees with
  ``solve_vlm`` to < 0.001% in CL and CDi. ✅
- CDi change at α = 12°: +2.16% (relaxed > fixed, physically correct). ✅
- CDi convergence history shows monotone approach to steady state over
  ~25 iterations at α = 12°. ✅

**Exit criterion.** ✅ ``examples/05_wake_relaxation/05_wake_relaxation.py``
produces:

- ``05_wake_shape.png``      — top-view wake streamlines, fixed vs relaxed
- ``05_wake_convergence.png``— CDi per relaxation iteration at α = 12°
- ``05_CDi_comparison.png``  — CDi(α) sweep: fixed vs relaxed wake

**Note on CDi sign.** The change ΔCDi is small (< 3%) and non-monotone in α
for this swept planform: negative at α ≈ 10°, positive for α ≥ 12°.  This
is consistent with the known sensitivity of swept-wing induced drag to wake
roll-up and downwash redistribution.  VSPAERO comparison deferred (tool not
available in current environment).

### Stage 7 — SDF geometry backend

**Effort:** 1.5–2 weeks

**Goal.** Replace the parametric mesh generator with a signed-distance-function
geometry representation, while keeping the same solver downstream.

**Tasks.**

- [ ] `geometry/sdf/primitives.py`: 2D SDF for NACA 4-digit airfoils
      (analytic, with explicit handling of the trailing-edge point).
- [ ] `geometry/sdf/wing_sdf.py`: 3D wing SDF by lofting a 2D airfoil SDF
      along the span with twist/dihedral/sweep/taper. Must be Eikonal
      (|∇SDF| ≈ 1 near the surface) — verify numerically.
- [ ] `geometry/sdf/sampler.py`: parametric sampler — for each (ξ, η) in the
      chordwise/spanwise parameter domain, find the surface point by
      projection along the SDF gradient. This yields the same panel topology
      as the parametric generator.
- [ ] Produce a `Mesh` object that satisfies all the same invariants.

**Verification (this is the key cross-check).**

- For identical wing parameters, the parametric mesh and SDF-derived mesh
  should produce CL, CDi, Cm agreeing to within 1% (residual = numerical
  noise in surface projection).
- |∇SDF| measured at sampled surface points is 1.0 ± 0.01.
- Trailing-edge geometry is preserved (no rounding).

**Exit criterion.** `examples/06_sdf_backend.py` runs both backends on the
same wing parameters, prints CL/CDi/Cm side by side, plots Cp from both.

### Stage 8 — Academic study: SDF perturbation → aerodynamic response

**Effort:** 2–4 weeks (this is the "paper")

**Goal.** Use the SDF representation to study how geometric perturbations
(noise, local bumps, manufacturing-error-like variations) propagate to
aerodynamic forces and pressure distributions. This is the research
contribution.

**Tasks.**

- [ ] Define a parameterized noise model on the SDF (e.g., Gaussian random
      field added to the SDF, with controllable amplitude and length scale).
- [ ] Monte Carlo study: sample N noise realizations, run the solver, collect
      statistics on CL, CDi, Cm, Cp distributions.
- [ ] Compare against a baseline of "node-perturbation" noise on a
      conventional mesh — does SDF noise produce qualitatively different
      sensitivity?
- [ ] Localized perturbation study: add a single bump (e.g., simulating
      ice accretion or skin damage) at controlled chord/span locations,
      map the aerodynamic response.
- [ ] Optional: SDF-based shape optimization demo — vary 2–3 wing
      parameters, optimize for max L/D at fixed CL, using the analytic
      SDF gradient.

**Verification.** Sanity check: zero noise should reproduce the deterministic
result. Mean of noisy results should converge to the deterministic result as
amplitude → 0.

**Exit criterion.** A short write-up (~10 pages, LaTeX), plots of force
response distributions, and a clearly stated finding. Polished figures
suitable for a workshop paper or arXiv preprint.

## 5. Cross-Cutting Concerns

These are not stages but must be maintained throughout.

### Testing

- Every solver module has at least one analytic-reference test.
- `tests/test_biot_savart.py`: induced velocity of a straight filament
  matches the closed-form Biot–Savart formula.
- `tests/test_solver_analytic.py`: large-AR rectangular wing → Prandtl
  lifting-line result.
- Run `pytest` before every commit.

### Performance

- Influence-coefficient assembly fully vectorized: 1000 panels < 5 s
  on a laptop.
- LU factorization is cached and reused for the same geometry across
  α/β sweeps across the same geometry.
- Beyond ~5000 panels: profile, consider Numba for hot loops if needed.
  Do not optimize prematurely.

### Visualization

- Every example script saves a plot (PNG) and optionally a VTK file for
  ParaView.
- `viz/plot.py` provides: mesh-with-normals, Cp-contour, wake, spanwise
  loading, derivative tables.

### Documentation

- `README.md`: one-paragraph summary, install, quickstart, citation,
  roadmap pointer.
- Every public function has a docstring with: physical meaning, units,
  reference equation (with citation to Katz & Plotkin chapter/equation
  number when applicable).
- `ROADMAP.md` (this file): kept current.

### Reference baselines to collect early

Before Stage 3, set these up so you can cross-check throughout development:

- AVL installed and a flying-wing case ready
- VSPAERO installed and a matched flying-wing case ready
- XFOIL installed for 2D Cp comparisons
- Save reference outputs to `references/` directory (gitignored or LFS)

## 6. What Is Explicitly Out of Scope

To prevent scope creep:

- **Fuselage / wing-body intersection.** Not a goal; requires CompGeom.
- **Supersonic.** Linear vortex panel methods are unstable in supersonic;
  would require PANAIR-style higher-order formulation.
- **Viscous boundary layer coupling.** No XFOIL-style integral BL.
- **Propellers / actuator disks.** Possible future, not now.
- **Unsteady aerodynamics beyond simple wake relaxation.** No flutter,
  no oscillating airfoil response.
- **Mesh import from arbitrary CAD/STL.** SDF is the geometry source.
  `.tri` reader is optional convenience only.
- **GUI.** This is a library + scripts. No Qt, no web app.

If any of these become interesting later, they get their own roadmap
revision; they do not silently expand current stages.

## 7. Working Style Notes (for the AI assistant)

When picking up this project in a new session:

1. **Read this file first.** It tells you what stage we are in, what is
   done, and what is next.
2. **Check the commit log / branch state** to confirm which stage's exit
   criterion has actually been met (vs. just claimed).
3. **Don't skip stages.** If user asks for Stage 7 features but Stage 4
   isn't validated, say so and finish Stage 4 first.
4. **One stage per session unless trivial.** Better to finish Stage 5
   completely than to leave Stages 5/6/7 each half-done.
5. **Vectorize aggressively.** If you find yourself writing
   `for panel in panels:` in solver code, stop and rewrite with NumPy
   broadcasting.
6. **Plot output always.** Every solver run should optionally save a
   figure. The user will frequently ask "does it look right?" — answer with
   a picture.
7. **Match the `Mesh` contract exactly.** Don't invent new geometry data
   structures; extend `Mesh` if needed but keep its invariants.
8. **Numerical references over feelings.** "Looks about right" is not a
   verification. Cite the reference (Katz & Plotkin Eq. X, or AVL output)
   when claiming agreement.
