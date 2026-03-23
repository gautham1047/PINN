"""
Animation utilities for visualizing 2D PDE solutions and velocity fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import MaxNLocator
from grid import Grid_2D

# Animation settings
DURATION = 5  # seconds
REPEAT_DELAY = 100000  # ms
CMAP = 'Wistia'
VELOCITY_CMAP = 'viridis'

def gen_anim(data: np.ndarray, grid: Grid_2D, file_name: str,
             z_label: str = "u", duration: float = DURATION) -> None:
    t_points = len(data)
    animation_interval = 1000 * duration / t_points  # t_points = frames

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

    max_z = np.max(data)
    min_z = np.min(data)

    def animate(i):
        ax.cla()
        trisurf = ax.plot_trisurf(grid.xv.flatten(), grid.yv.flatten(), data[i-1].flatten(),
                                  cmap=plt.get_cmap(CMAP))
        ax.set_xlim((grid.x_grid.x_i, grid.x_grid.x_f))
        ax.set_ylim((grid.y_grid.x_i, grid.y_grid.x_f))
        ax.set_zlim((min_z, max_z))

        # Set axis labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(z_label)

    # Formatting the graph
    ax.view_init(25, 45, 0)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=t_points,
                                  interval=animation_interval, blit=False,
                                  repeat_delay=REPEAT_DELAY)
    anim.save(file_name)
    plt.close(fig)


def gen_velocity_anim(velocity_data: np.ndarray, grid: Grid_2D, file_name: str,
                     duration: float = DURATION) -> None:
    t_points = len(velocity_data)
    animation_interval = 1000 * duration / t_points  # t_points = frames

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

    max_z = np.max(velocity_data)
    min_z = np.min(velocity_data)

    def animate(i):
        ax.cla()
        trisurf = ax.plot_trisurf(grid.xv.flatten(), grid.yv.flatten(),
                                  velocity_data[i-1].flatten(),
                                  cmap=plt.get_cmap(VELOCITY_CMAP))
        ax.set_xlim((grid.x_grid.x_i, grid.x_grid.x_f))
        ax.set_ylim((grid.y_grid.x_i, grid.y_grid.x_f))
        ax.set_zlim((min_z, max_z))

        # Set axis labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("du/dt")

    # Formatting the graph
    ax.view_init(25, 45, 0)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=t_points,
                                  interval=animation_interval, blit=False,
                                  repeat_delay=REPEAT_DELAY)
    anim.save(file_name)
    plt.close(fig)
