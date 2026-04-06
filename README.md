# Irregular-Domain 2D PDE Solver

A finite-difference solver for parabolic (heat) and hyperbolic (wave) PDEs on non-rectangular 2D domains, using embedded boundary methods on Cartesian grids.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="README/solution.gif" alt="Vibrating drum" width="400"/>
        <br/>
        <em>Vibrating drum</em>
      </td>
      <td align="center">
        <img src="README/velocity.gif" alt="Vibrating drum velocity" width="400"/>
        <br/>
        <em>Vibrating drum velocity</em>
      </td>
    </tr>
  </table>
</div>

## Features

- **Irregular domains** via embedded Dirichlet mask boundaries (no mesh conforming required)
- **Two wave-equation paths**: leapfrog with Kreiss–Petersson (K-P) ghost-point elimination (recommended) and legacy Newmark
- **Parabolic solvers**: forward Euler and RK4
- **Symbolic PDE specification** via SymPy
- **Animation output**: 3D surface and 2D heatmap backends, MP4/GIF, with time and spatial stride

---

## File Overview

### [`fd_coeffs.py`](solver_lib/fd_coeffs.py) — FD Coefficient Helpers

Computes finite-difference coefficients from arbitrary stencils.

| Function | Description |
|---|---|
| `calculate_fd_coefficients(stencil, derivative_order)` | Vandermonde-based solver for arbitrary stencils |
| `forward_difference()`, `central_difference()`, `backward_difference()` | Classic 1st/2nd-order FD generators |
| `neumann_boundary_forward/backward(h, accuracy_order)` | Neumann BC enforcement at domain edges |
| `dirichlet_pseudo_boundary_forward/backward(alpha_over_h, d, n)` | Fractional-step stencils for Dirichlet pseudo-boundary points |
| `lagrange_weights(x_nodes, x_query)` | 3-point quadratic Lagrange interpolation weights |
| `normal_lagrange_weights(xi_gamma, xi_I)` | Normal-direction weights for K-P embedded BC method |

---

### [`grid.py`](solver_lib/grid.py) — Grid and Derivative Matrices

#### `Grid_1D`
Uniform 1D grid. Key methods:
- `initialize_values(func, x_symbol)` — evaluate a symbolic function onto grid nodes
- `derivative_matrix(order, accuracy_order, strategy)` — build FD derivative matrix; `strategy` is `'forward_central_backward'` or `'custom_stencil'`
- `set_boundary_dirichlet()`, `set_boundary_neumann()`

#### `Grid_2D`
2D grid on `[x_i, x_f] × [y_i, y_f]`. Owns `x_grid`, `y_grid` (`Grid_1D` objects), meshgrids `xv`/`yv`, and a flat `values` array (row-major). Key methods:
- `initialize_values(func, x_symbol, y_symbol)`
- `derivative_matrix(order, direction)` — `direction ∈ {'xy'` (Laplacian via Kronecker product), `'x'`, `'y'}`
- `set_boundary_dirichlet(x_0_func, x_L_func, y_0_func, y_L_func, time=...)` — time-dependent BCs supported
- `apply_boundary_conditions(bc, ...)` — unified BC dispatch
- `apply_dirichlet_mask(bc, n=1)` — enforce geometric mask BC; warms K-P pseudo-boundary cache

#### `VelocityGrid(Grid_2D)`
Adds a `velocity` flat array. Required by `solve_newmark()`; **deprecated for the leapfrog path**.

---

### [`boundary_conditions.py`](solver_lib/boundary_conditions.py) — BC Specification

#### `BoundaryConditions`
Axis-aligned Dirichlet/Neumann on 4 rectangle edges. Detects time-dependence automatically via `inspect.signature`.

#### `DirichletMask`
Geometric mask BC for irregular domains. `mask_function(grid)` returns a bool array (`True` = outside domain).
- `dirichlet_value`, `velocity_value` — enforced on masked points
- **Heat path** (linear Collatz method):
  - `get_pseudo_boundary_points(grid, n=1)` — finds interior points adjacent to boundary; returns fractional boundary distances `α`
  - `compute_pseudo_boundary_derivative(grid, flat_idx, direction, d, n=1)` — builds local fractional-step stencil
- **Wave path** (K-P method):
  - `preprocess_kp(grid, c_sq_values, gamma=0.25)` — classifies grid points (interior / near-boundary / ghost / exterior), algebraically eliminates ghost dofs by substituting normal-direction Lagrange interpolants, and returns `(A, b_func, interior_mask, interior_to_full, full_to_interior)`. Result is cached in `_kp_cache`.

> **K-P method:** Ghost values are expressed as linear combinations of interior dofs and BC values using quadratic Lagrange interpolation along the inward normal. This yields a provably O(Δx²) scheme stable under CFL = Δt/h < 1/√2 (with γ ≥ 0.25). Inspired by Kreiss & Petersson (2006), *SIAM J. Sci. Comput.* 27(4), 1141–1167.
>
> **Linear Collatz (heat):** O(h) local truncation error at pseudo-boundary rows → O(Δx²) global solution error (dominated by boundary). Upgrade to Shortley–Weller for better accuracy.

---

### [`differential_equation.py`](solver_lib/differential_equation.py) — Symbolic PDE

#### `DifferentialEquation`
Parses a SymPy RHS expression for a linear PDE.
- `__init__(rhs, u_symbol, x_symbol, y_symbol, t_symbol, lhs=None, time_derivative_order=1)`
- `extract_spatial_terms()` — returns `{(dx_order, dy_order): coeff_expr, 'source': source_expr}`
- `get_coefficient(dx_order, dy_order)`, `get_source_term()`
- `is_parabolic`, `is_hyperbolic` — properties

#### Factory functions

Instead of subclasses, built-in PDEs are created via factory functions that return `(equation, (x, y, t, u))`:

```python
equation, (x, y, t, u) = HeatEquation(alpha=0.05)
# du/dt = alpha * (d²u/dx² + d²u/dy²)

equation, (x, y, t, u) = WaveEquation(c=1.0, gamma=0.0)
# d²u/dt² + gamma·du/dt = c²(d²u/dx² + d²u/dy²)
```

The returned symbols `(x, y, t, u)` are the SymPy objects used to write initial conditions and source terms. Discard with `_` any symbols you don't need:

```python
equation, (x, y, _, _) = HeatEquation(alpha=0.1)
initial_u = sin(pi * x) * sin(pi * y)
```

---

### [`solver.py`](solver_lib/solver.py) — Time-Stepping

#### `Solver`
`Solver(equation, grid, bc, t_i, t_f, t_points, initial_condition, initial_velocity=None)`

**Parabolic:**
- `solve_euler()` — forward Euler: `u^{n+1} = u^n + Δt · du/dt`
- `solve_rk4()` — 4th-order Runge–Kutta

**Hyperbolic:**
- `solve_leapfrog(gamma=0.25)` — explicit leapfrog using the precomputed K-P matrix:
  `u^{n+1} = 2u^n − u^{n-1} + Δt² [A u^n + b(t_n) + F(t_n)]`
  Requires `DirichletMask` and time-independent c². Uses `u^{-1} = u^0 − Δt·u1` for 2nd-order startup.
- `solve_newmark(beta, gamma)` — implicit Newmark; requires `VelocityGrid`. Tracks explicit velocity history. *(Legacy)*

**Accessors:**
- `get_solution_2d(step)`, `get_solution_at_time(idx)`
- `get_velocity_2d(step)`, `get_velocity_at_time(idx)`
- `reset()`

**Animation:**
```python
solver.animate(file_name, z_label="u", duration=5.0,
               output_type="3D",   # "3D" trisurf or "2D" heatmap
               stride=1,           # render every nth time step
               spatial_stride=1,   # sample every nth grid point
               output_params=None) # ffmpeg flags for .mp4

solver.animate_velocity(file_name, ...)  # same signature, no z_label
```

**Internal helpers:**
- `compute_dudt(u_values, time)` — evaluates PDE RHS by summing coefficient × derivative-matrix products + source
- `_extract_c_sq(grid)` — extracts wave speed field c²(x,y)
- `_validate_c_sq_time_independence()` — enforces leapfrog precondition

---

### [`animate.py`](solver_lib/animate.py) — Animation Backends

Two backends are available. Both support `.gif` and `.mp4` output; MP4 uses H.264 via ffmpeg (requires `imageio-ffmpeg`).

#### Matplotlib 3-D (`gen_anim`, `gen_velocity_anim`)
Renders a `plot_trisurf` surface at each frame via `FuncAnimation`. Gives a 3-D perspective view. Slower for large grids due to full geometry rebuild per frame.

#### Fast 2-D (`gen_anim_fast`, `gen_velocity_anim_fast`)
Renders a 2-D heatmap using `imshow`. The figure is built once; only pixel data is updated per frame, so matplotlib's layout overhead runs exactly once. Significantly faster than the 3-D backend.

- Colormap: `seismic` (blue→white→red) for displacement — diverging palette makes the sign of oscillation immediately visible
- Colormap: `RdYlBu_r` for velocity

#### Common parameters

| Parameter | Description |
|---|---|
| `file_name` | Output path — `.gif` or `.mp4` |
| `duration` | Total animation length in seconds |
| `stride` | Render every nth time step (e.g. `stride=2` halves frame count) |
| `spatial_stride` | Sample every nth grid point in x and y (3-D backend also supported) |
| `output_params` | ffmpeg argument list for `.mp4`; default `["-preset", "ultrafast", "-crf", "23"]` |

#### Speed benchmark

```bash
cd solver_lib
python test_animation_speed.py --nx 30 --ny 30 --frames 50
```

Expected ordering fastest→slowest: `fast (mp4)` → `fast (gif)` → `matplotlib (mp4)` → `matplotlib (gif)`

---

## Example Scripts

### [`vibrating_drum.py`](demo/vibrating_drum.py)
Undamped circular drum, radius 1, on a high-resolution grid.
- Wave speed c = 1, time span [0, 5], K-P leapfrog solver
- IC: Gaussian bump, zero initial velocity
- BC: `DirichletMask` (zero on circle boundary)
- Output: MP4 in `demo/vibrating_drum/` (3-D and 2-D)

### [`vibrating_drum_damped.py`](demo/vibrating_drum_damped.py)
Damped wave equation `u_tt + 0.5·u_t = ∇²u` on the same domain.
- Solver: `solve_newmark(beta=0.25, gamma=0.5)` with `VelocityGrid`
- Outputs displacement + velocity animations

### [`heat_equation_dirichlet.py`](demo/heat_equation_dirichlet.py)
Heat equation on `[0,1]²` with zero Dirichlet BCs on all edges.
- IC: `sin(πx)sin(πy)` — exact solution decays as `exp(−2π²αt)`
- Solver: RK4

### [`heat_equation_neumann.py`](demo/heat_equation_neumann.py)
Heat equation with mixed BCs: Dirichlet on x-edges (`u=0` at x=0, `u=1` at x=1), Neumann on y-edges.
- IC: linear profile `x` plus a y-localised Gaussian bump
- Solver: RK4

---

## Architecture Notes

### Two Wave-Equation Paths

| | Leapfrog (K-P) | Newmark (legacy) |
|---|---|---|
| Grid | `Grid_2D` | `VelocityGrid` |
| Boundaries | `DirichletMask` (arbitrary curved) | `BoundaryConditions` (axis-aligned only) |
| Wave speed | Must be time-independent | Can vary |
| Damping | Not supported | Supported |
| Velocity history | Not tracked | Tracked |
| Accuracy | O(Δx², Δt²), CFL < 1/√2 | Unconditionally stable (β=0.25) |

### CFL Condition (Leapfrog)
Stability requires `Δt/h < 1/√2 ≈ 0.707`. With γ = 0.25, the K-P ghost-point stabilization ensures this extends to cells cut by curved boundaries.
