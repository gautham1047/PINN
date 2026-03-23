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
        <img src="README/velocity.gif" alt="Vibrating drum (damped)" width="400"/>
        <br/>
        <em>Vibrating drum (damped)</em>
      </td>
    </tr>
  </table>
</div>

## Features

- **Irregular domains** via embedded Dirichlet mask boundaries (no mesh conforming required)
- **Two wave-equation paths**: leapfrog with Kreiss–Petersson (K-P) ghost-point elimination (recommended) and legacy Newmark
- **Parabolic solvers**: forward Euler and RK4
- **Symbolic PDE specification** via SymPy
- **Animation output** (3D surface, MP4/GIF)

---

## File Overview

### [`fd_coeffs.py`](fd_coeffs.py) — FD Coefficient Helpers

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

### [`grid.py`](grid.py) — Grid and Derivative Matrices

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

### [`boundary_conditions.py`](boundary_conditions.py) — BC Specification

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

### [`differential_equation.py`](differential_equation.py) — Symbolic PDE

#### `DifferentialEquation`
Parses a SymPy RHS expression for a linear PDE.
- `__init__(rhs, u_symbol, x_symbol, y_symbol, t_symbol, lhs=None, time_derivative_order=1)`
- `extract_spatial_terms()` — returns `{(dx_order, dy_order): coeff_expr, 'source': source_expr}`
- `get_coefficient(dx_order, dy_order)`, `get_source_term()`
- `is_parabolic`, `is_hyperbolic` — properties

#### `WaveEquation(DifferentialEquation)`
Convenience subclass: `WaveEquation(c, gamma=0)` builds `u_tt + γ·u_t = c^2 ∇^2 u`.

---

### [`solver.py`](solver.py) — Time-Stepping

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
- `animate()`, `animate_velocity()`
- `reset()`

**Internal helpers:**
- `compute_dudt(u_values, time)` — evaluates PDE RHS by summing coefficient × derivative-matrix products + source
- `_extract_c_sq(grid)` — extracts wave speed field c²(x,y)
- `_validate_c_sq_time_independence()` — enforces leapfrog precondition

---

### [`animate.py`](animate.py) — Animation

- `gen_anim(data, grid, file_name, z_label="u", duration=5.0)` — 3D `plot_trisurf` animation over solution history; colormap `'Wistia'`
- `gen_velocity_anim(velocity_data, grid, file_name, duration=5.0)` — same for velocity; colormap `'viridis'`

---

## Example Scripts

### [`vibrating_drum.py`](vibrating_drum.py)
Undamped circular drum, radius 1, on a 40×40 grid.
- Wave speed c = 1, time span [0, 5], 300 steps (CFL ≈ 0.75 < 1/√2 ✓)
- IC: Gaussian bump `exp(−(x²+y²)/(2·0.3²))`, zero initial velocity
- BC: `DirichletMask` (zero on circle boundary)
- Solver: `solve_leapfrog(gamma=0.25)`
- Output: `vibrating_drum/vibrating_drum_solution.gif`

### [`vibrating_drum_damped.py`](vibrating_drum_damped.py)
Damped wave equation `u_tt + 0.5·u_t = ∇²u` on same domain.
- Solver: `solve_newmark(beta=0.25, gamma=0.5)` with `VelocityGrid`
- Outputs displacement + velocity animations: `vibrating_drum/vibrating_drum_damped_*.gif`

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
