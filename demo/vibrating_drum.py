import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver_lib'))

from sympy import exp
from grid import Grid_2D
from boundary_conditions import DirichletMask
from differential_equation import WaveEquation
from solver import Solver

out_dir = os.path.join(os.path.dirname(__file__), 'vibrating_drum')
os.makedirs(out_dir, exist_ok=True)

c = 1.0
drum_radius = 1.0

# Grid parameters
x_i, x_f = (-drum_radius, drum_radius)
y_i, y_f = (-drum_radius, drum_radius)
t_i, t_f = (0, 5.0)

s = 3

x_points = 40 * s
y_points = 40 * s
t_points = 500 * s

equation, (x, y, _, _) = WaveEquation(c)

grid = Grid_2D(x_points, y_points, x_i, x_f, y_i, y_f,
               accuracy_order=2, strategy='custom_stencil')

def drum_boundary_mask(grid, radius):
    return grid.xv**2 + grid.yv**2 >= radius**2

bc = DirichletMask(
    mask_function=lambda g: drum_boundary_mask(g, drum_radius),
    dirichlet_value=0.0
)

initial_amplitude = 1.0
width = 0.3

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

h = grid.x_grid.delta
cfl = solver.t_delta / h
print(f"CFL = Δt/h = {cfl:.4f}  (must be < {1/np.sqrt(2):.4f} for stability)")
if cfl >= 1.0 / np.sqrt(2):
    print("bad CFL condition - Reduce t_points or increase grid resolution.")

# Solve using K-P leapfrog
solution = solver.solve_leapfrog(gamma=0.25)

print("animating...")

from time import time

start = time()
solver.animate(f'{out_dir}/solution2.mp4', stride=4 * s, spatial_stride=s)
print(f"3D mp4 done in {time() - start:.2f} seconds")
start = time()
solver.animate(f'{out_dir}/solution2_2d.mp4', output_type="2D", stride=4 * s, spatial_stride=s)
print(f"2D mp4 done in {time() - start:.2f} seconds")