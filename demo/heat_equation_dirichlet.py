import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver_lib'))

from sympy import sin, pi
from grid import Grid_2D
from boundary_conditions import BoundaryConditions
from differential_equation import HeatEquation
from solver import Solver

out_dir = os.path.join(os.path.dirname(__file__), 'heat_equation_dirichlet')
os.makedirs(out_dir, exist_ok=True)

# Physical parameters
alpha = 0.05   # thermal diffusivity

# Grid parameters
x_i, x_f = 0.0, 1.0
y_i, y_f = 0.0, 1.0
t_i, t_f = 0.0, 2.0

x_points = 25
y_points = 25
t_points = 200

equation, (x, y, _, _) = HeatEquation(alpha=alpha)

grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f, accuracy_order=2, strategy='custom_stencil')

# Dirichlet BC: u = 0 on all edges (default)
bc = BoundaryConditions()

# IC: sin(pi*x)sin(pi*y) — satisfies the zero Dirichlet BCs exactly.
# Exact solution: u(x,y,t) = exp(-2pi^2 * alpha * t) * sin(pi * x)sin(pi * y)
initial_u = sin(pi * x) * sin(pi * y)

solver = Solver(
    equation=equation,
    grid=grid,
    boundary_conditions=bc,
    t_i=t_i, t_f=t_f, t_points=t_points,
    initial_condition=initial_u,
)

solution = solver.solve_rk4()

print("animating...")
solver.animate(f'{out_dir}/solution.gif')
