# LSPAero — Development Roadmap & Task List

**A level-set panel method for flying-wing aerodynamics in Python.**

This document is the working contract between the developer and the AI coding
assistant. It defines what is being built, in what order, and how each step is
verified. Update it as the project evolves; treat it as living, not historical.

---

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

---

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
6. **One thing at a time.** Do not start stability derivatives while wake
   relaxation is half-debugged. Finish, verify, commit, then move on.

---

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
│   │   ├── forces.py         # Cp, CL, CDi, Cm integration
│   │   ├── stability.py      # finite-difference derivatives
│   │   └── controls.py       # control-surface deflection (equivalent BC)
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
│   ├── 05_stability_derivatives/
│   ├── 06_control_surface/
│   ├── 07_wake_relaxation/
│   ├── 08_sdf_backend/
│   └── 09_sdf_perturbation_study/
└── tests/
    ├── test_biot_savart.py
    ├── test_mesh.py
    ├── test_solver_analytic.py
    ├── test_pg.py
    ├── test_stability.py
    └── test_sdf.py
```

---

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

---

## 4. Stage-by-Stage Roadmap

Each stage has: **goal · inputs · outputs · verification · exit criterion**.
Estimated effort is for focused work; multiply by 1.5–2 for learning-while-doing.

---

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

---

### Stage 2 — Mesh data structure & parametric flying-wing generator
**Effort:** 4–6 days

**Goal.** The `Mesh` class and a generator that turns
(span, root chord, tip chord, sweep, dihedral, twist, airfoil) into a valid
`Mesh` object.

**Tasks.**
- [ ] `geometry/mesh.py`: `Mesh` class with invariants and self-check.
- [ ] `geometry/wing.py`: parametric wing builder:
      - cosine spanwise distribution
      - cosine chordwise distribution
      - twist + dihedral + sweep + taper
      - tip cap (closed surface)
      - trailing-edge pair identification
- [ ] `viz/plot.py`: render mesh with normals as arrows, highlight TE in red.
- [ ] Mesh validity tests (`tests/test_mesh.py`): watertightness, normal
      orientation, TE pairs.

**Verification.**
- Plot a rectangular wing, swept wing, and tapered swept-twisted flying wing.
  All look right.
- `Mesh.is_watertight()` returns True.
- All normals point outward (verified by inflating geometry by ε along normals
  and checking volume increases).

**Exit criterion.** Three meshes saved as VTK files, viewable in ParaView,
with normals and TE pairs visible.

---

### Stage 3 — VLM solver (thin surface, fixed wake)
**Effort:** 1–1.5 weeks

**Goal.** A working vortex-lattice solver on the mean camber surface,
producing CL, CDi, Cm and spanwise loading.

**Tasks.**
- [ ] `solver/biot_savart.py`: vectorized vortex-segment induced velocity.
      Uses NumPy broadcasting; no Python loops over panels.
      Cutoff radius for near-singularity.
- [ ] `solver/influence.py`: assemble AIC matrix
      A[i,j] = (induced velocity of ring j at collocation i) · n_i.
- [ ] `solver/kutta.py`: trailing-edge condition (last chordwise panel sheds
      wake with Γ_panel == Γ_wake).
- [ ] `solver/wake.py`: fixed wake aligned with freestream direction.
- [ ] `solver/solve.py`: assemble RHS, solve linear system, return circulations.
- [ ] `physics/forces.py`: Kutta–Joukowski force per filament,
      integrate to CL, CDi (Trefftz-plane or near-field), Cm about a reference
      point.

**Verification.**
- Rectangular wing AR = 8: CL_α within 2% of Prandtl lifting-line.
- Elliptic wing: CDi within 2% of CL²/(πAR).
- Swept-wing case: agreement with AVL within a few percent.

**Exit criterion.** `examples/01_rectangular_vlm.py` and
`examples/02_swept_flying_wing_vlm.py` produce plots and printouts matching
references.

---

### Stage 4 — Thick-surface vortex-panel solver
**Effort:** 1.5–2 weeks

**Goal.** Upgrade from VLM to thick-panel method. Now we compute **real
surface Cp**, not just ΔCp. This is where VSPAERO's Panel mode lives.

**Tasks.**
- [ ] Extend `geometry/wing.py`: thick mesh with upper and lower surfaces,
      tip cap, proper TE pair identification across the thick TE.
- [ ] Reuse `solver/biot_savart.py` and `influence.py` unchanged — they
      operate on a generic `Mesh`.
- [ ] Update `solver/kutta.py`: thick-TE Kutta as upper/lower panel
      circulation jump, wake sheds the jump.
- [ ] `physics/forces.py`: compute surface tangential velocity from circulation
      gradient; apply Bernoulli → Cp; integrate over surface for forces.
- [ ] Plot upper/lower Cp distribution at multiple spanwise stations.

**Verification.**
- Same flying-wing geometry as Stage 3: CL agrees with VLM result.
- Cp distribution shape qualitatively matches XFOIL 2D at midspan section
  (we won't expect perfect agreement; this is 3D inviscid).
- Compare against VSPAERO thick-panel for the same geometry — within a few
  percent on CL, CDi, Cm.

**Exit criterion.** `examples/03_thick_flying_wing.py` produces Cp contour
on the wing surface (PyVista) and chordwise Cp plots at several stations.

---

### Stage 5 — Prandtl–Glauert compressibility correction
**Effort:** 1–2 days

**Goal.** Support subsonic compressible flow (up to about M = 0.7) via PG.

**Tasks.**
- [ ] `physics/pg_correction.py`:
      - stretch geometry in freestream direction by 1/β where β = √(1 − M²)
      - solve in PG-transformed space
      - rescale Cp by 1/β
      - reference geometric quantities (area, MAC) computed in the **original**
        (physical) frame
- [ ] Add Mach number as a solver input.

**Verification.** At M = 0.5 on a baseline flying wing, Cp should scale
approximately as the incompressible Cp divided by β. Total CL scales the same.
Compare against VSPAERO with same Mach number.

**Exit criterion.** `examples/04_compressibility.py` plots CL vs Mach at fixed
α, showing Glauert factor behavior.

---

### Stage 6 — Stability derivatives
**Effort:** 4–7 days

**Goal.** Compute CLα, CLβ, Cmα, Cnβ, Clβ, and rotary derivatives CLq, Cmq,
Cnp, Clp, Cnr, Clr by finite difference about a trim state.

**Tasks.**
- [ ] `physics/stability.py`:
      - α, β perturbations: tilt the freestream vector
      - p, q, r perturbations: add rotational velocity field at each
        collocation point, **not** by rotating the mesh
      - reuse the LU-factored AIC matrix (matrix doesn't change for these
        perturbations — only the RHS does) for huge speedup
      - non-dimensionalize correctly (chord, span, V∞ in the standard
        convention)
- [ ] Output a clean derivative table.

**Verification.** For a swept flying wing, compare the full derivative table
against AVL. Agreement within a few percent on all entries is expected;
larger discrepancies on rotary derivatives are normal but should be
explainable.

**Exit criterion.** `examples/05_stability_derivatives.py` prints a table
matching AVL output side by side.

---

### Stage 7 — Control surfaces (equivalent boundary condition)
**Effort:** 5–7 days

**Goal.** Deflect elevons (a flying wing's primary control surface) by
modifying the no-penetration boundary condition, without re-meshing.

**Tasks.**
- [ ] Add `control_surface` metadata to `Mesh`: which panels belong to
      which control surface, hinge axis location and direction.
- [ ] `physics/controls.py`:
      - effective normal: rotate the panel's normal by the deflection angle
        about the hinge axis
      - this only changes the RHS of the linear system; AIC matrix unchanged
      - LU factorization reused across deflections — fast sweeps
- [ ] Support multiple simultaneous control surfaces (elevon left/right
      independently) for roll + pitch.

**Verification.**
- Elevon deflection of +10° produces a Cm change of the expected sign and
  rough magnitude (compare against AVL).
- Differential elevon (one up, one down) produces a Cl roll moment.
- Sweep of deflection from −20° to +20° shows linear behavior, as expected
  for a potential-flow model.

**Exit criterion.** `examples/06_control_surface.py` produces a plot of
Cm vs δ_elevon and a 3D view of the deflected wing.

---

### Stage 8 — Wake relaxation (simplified)
**Effort:** 1–2 weeks (timebox strictly)

**Goal.** Allow the wake to align with the local flow direction, improving
induced drag prediction at moderate-to-high α.

**Tasks.**
- [ ] `solver/wake.py`: extend to support flexible wake nodes.
- [ ] Iteration scheme:
      - solve with current wake
      - compute velocity at each wake node from current solution
      - advect nodes downstream along streamlines for a fixed pseudo-time
      - re-solve
      - repeat until wake-node displacement < tolerance
- [ ] Stabilization: under-relaxation factor (start at 0.3),
      Lamb–Oseen vortex core for self-induced wake velocities.
- [ ] First version: only relax wake nodes beyond N filament-lengths
      downstream of the TE — protects the Kutta condition.

**Verification.**
- Fixed-wake CDi at α = 10° on AR=6 swept wing.
- Relaxed-wake CDi at the same condition.
- The relaxed result should be slightly higher than the fixed result, and
  closer to higher-fidelity references (VSPAERO with relaxation, or
  experiment if available).

**Exit criterion.** `examples/07_wake_relaxation.py` shows the wake shape
before/after relaxation and reports CDi for both. Convergence history of
wake node displacement is plotted.

**Fallback.** If relaxation refuses to converge within the timebox, ship
"quasi-rigid wake": wake aligned with the local freestream direction at the
TE for each strip (varies spanwise but doesn't iterate). This gets 80% of
the benefit with no convergence risk.

---

### Stage 9 — SDF geometry backend
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

**Exit criterion.** `examples/08_sdf_backend.py` runs both backends on the
same wing parameters, prints CL/CDi/Cm side by side, plots Cp from both.

---

### Stage 10 — Academic study: SDF perturbation → aerodynamic response
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

---

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
  control deflections, α/β sweeps, and stability-derivative perturbations.
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

---

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

---

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
