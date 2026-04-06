import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver_lib'))

from sympy import exp, sin, pi
from grid import Grid_2D
from boundary_conditions import BoundaryConditions
from differential_equation import HeatEquation
from solver import Solver
import numpy as np

out_dir = os.path.join(os.path.dirname(__file__), 'heat_equation_neumann')
os.makedirs(out_dir, exist_ok=True)

# Physical parameters
alpha = 0.05   # thermal diffusivity

# Grid parameters
x_i, x_f = 0.0, 1.0
y_i, y_f = 0.0, 1.0
t_i, t_f = 0.0, 5.0

x_points = 25
y_points = 25
t_points = 500

equation, (x, y, _, _) = HeatEquation(alpha=alpha)

grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f, accuracy_order=2, strategy='custom_stencil')

# Mixed BCs: Dirichlet on x edges (u=0 at x=0, u=1 at x=1), Neumann (du/dn=0) on y edges.
bc = BoundaryConditions(
    x_0_is_dirichlet=True,
    x_L_is_dirichlet=True,
    y_0_is_dirichlet=False,
    y_L_is_dirichlet=False,
    x_0_func=lambda coords: np.zeros_like(coords),
    x_L_func=lambda coords: np.ones_like(coords),
)

# IC: linear profile x (satisfies Dirichlet BCs) plus a y-localised bump.
# sin(pi*x) vanishes at both x=0 and x=1, so the BCs are preserved exactly.
width = 0.2
initial_u = x + sin(pi * x) * exp(-((y - 0.5)**2) / (2 * width**2))

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
