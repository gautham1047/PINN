import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver_lib'))

from sympy import exp
from grid import VelocityGrid
from boundary_conditions import DirichletMask
from differential_equation import WaveEquation
from solver import Solver

# Physical parameters
c = 1.0      
drum_radius = 1.0   

# Grid parameters
x_i, x_f = (-drum_radius, drum_radius)
y_i, y_f = (-drum_radius, drum_radius)
t_i, t_f = (0, 5.0)

x_points = 40
y_points = 40
t_points = 300

out_dir = os.path.join(os.path.dirname(__file__), 'vibrating_drum_damped')
os.makedirs(out_dir, exist_ok=True)

equation, (x, y, _, _) = WaveEquation(c=c, gamma=0.5)  # dampening coeff = 0.5

# Create grid
grid = VelocityGrid(x_points, y_points, x_i, x_f, y_i, y_f, accuracy_order=2, strategy='custom_stencil')

# circular boundary condition
def drum_boundary_mask(grid, radius):
    xv = grid.xv
    yv = grid.yv
    r_squared = xv**2 + yv**2
    return r_squared >= radius**2

bc = DirichletMask(
    mask_function=lambda grid: drum_boundary_mask(grid, drum_radius),
    dirichlet_value=0.0
)

# initial condition
initial_amplitude = 1.0
width = 0.3  # Controls how sharp the bump is

initial_u = initial_amplitude * exp(-((x**2 + y**2) / (2 * width**2)))
initial_v = 0

solver = Solver(
    equation=equation,
    grid=grid,
    boundary_conditions=bc,
    t_i=t_i, t_f=t_f, t_points=t_points,
    initial_condition=initial_u,
    initial_velocity=initial_v,
)

solution = solver.solve_newmark(beta=0.25, gamma=0.5)

print("animating...")

solver.animate(f'{out_dir}/solution.gif')
solver.animate_velocity(f'{out_dir}/velocity.gif')