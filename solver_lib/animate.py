"""
Animation utilities for visualizing 2D PDE solutions and velocity fields.

Backends
--------
matplotlib
    3-D trisurf via FuncAnimation. GIF uses the Pillow writer; MP4 uses
    ffmpeg (requires imageio-ffmpeg or a system ffmpeg on PATH).

fast (recommended)
    2-D imshow + direct imageio frame writes. No 3-D geometry; only pixel
    data changes per frame. Fastest for typical grid sizes.
    MP4 uses H.264 ultrafast preset by default.

Common parameters
-----------------
stride : int
    Time stride — render every nth time step (default 1 = all frames).
    stride=2 halves the frame count and roughly halves render time.
spatial_stride : int
    Spatial stride — sample every nth grid point in x and y (default 1).
    spatial_stride=2 reduces the point count to ~25% of the original.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import MaxNLocator
from grid import Grid_2D
import imageio

# Animation settings
DURATION = 5  # seconds
REPEAT_DELAY = 100000  # ms
CMAP = 'Wistia'
VELOCITY_CMAP = 'viridis'

# 2-D fast backend colormaps — diverging palettes make oscillation sign visible
FAST_CMAP = 'seismic'           # blue→white→red; clearly shows +/- displacement
FAST_VELOCITY_CMAP = 'RdYlBu_r' # red→yellow→blue; high contrast for velocity

FFMPEG_FAST = ["-preset", "ultrafast", "-crf", "23"]


def _is_mp4(file_name: str) -> bool:
    return file_name.rsplit(".", 1)[-1].lower() == "mp4"


def _imageio_kwargs(file_name: str, fps: float, output_params: list | None) -> dict:
    """Build imageio.get_writer kwargs for GIF or MP4."""
    kwargs: dict = {"fps": fps}
    if _is_mp4(file_name):
        kwargs["codec"] = "libx264"
        kwargs["output_params"] = output_params if output_params is not None else FFMPEG_FAST
    return kwargs


# ---------------------------------------------------------------------------
# Matplotlib 3-D backend
# ---------------------------------------------------------------------------

def gen_anim(data: np.ndarray, grid: Grid_2D, file_name: str,
             z_label: str = "u", duration: float = DURATION,
             stride: int = 1, spatial_stride: int = 1,
             output_params: list | None = None) -> None:
    """3-D trisurf animation via FuncAnimation.

    Parameters
    ----------
    stride : int
        Render every nth time step.
    spatial_stride : int
        Sample every nth grid point in x and y.
    output_params : list
        Extra ffmpeg args for .mp4. Ignored for GIF. None → ultrafast preset.
    """
    frames = data[::stride]
    t_points = len(frames)
    fps = max(1, t_points / duration)
    animation_interval = 1000 * duration / t_points

    nx = grid.x_points
    ny = grid.y_points
    x_s = grid.xv[::spatial_stride, ::spatial_stride].flatten()
    y_s = grid.yv[::spatial_stride, ::spatial_stride].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

    max_z = np.max(frames)
    min_z = np.min(frames)

    def animate(i):
        ax.cla()
        z_s = frames[i-1].reshape(ny, nx)[::spatial_stride, ::spatial_stride].flatten()
        ax.plot_trisurf(x_s, y_s, z_s, cmap=plt.get_cmap(CMAP))
        ax.set_xlim((grid.x_grid.x_i, grid.x_grid.x_f))
        ax.set_ylim((grid.y_grid.x_i, grid.y_grid.x_f))
        ax.set_zlim((min_z, max_z))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(z_label)

    ax.view_init(25, 45, 0)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=t_points,
                                   interval=animation_interval, blit=False,
                                   repeat_delay=REPEAT_DELAY)

    if _is_mp4(file_name):
        extra = output_params if output_params is not None else FFMPEG_FAST
        anim.save(file_name, writer="ffmpeg", fps=fps, extra_args=extra)
    else:
        anim.save(file_name, writer="pillow", fps=fps)

    plt.close(fig)


def gen_velocity_anim(velocity_data: np.ndarray, grid: Grid_2D, file_name: str,
                      duration: float = DURATION, stride: int = 1,
                      spatial_stride: int = 1,
                      output_params: list | None = None) -> None:
    """3-D trisurf velocity animation via FuncAnimation. See gen_anim for details."""
    frames = velocity_data[::stride]
    t_points = len(frames)
    fps = max(1, t_points / duration)
    animation_interval = 1000 * duration / t_points

    nx = grid.x_points
    ny = grid.y_points
    x_s = grid.xv[::spatial_stride, ::spatial_stride].flatten()
    y_s = grid.yv[::spatial_stride, ::spatial_stride].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

    max_z = np.max(frames)
    min_z = np.min(frames)

    def animate(i):
        ax.cla()
        z_s = frames[i-1].reshape(ny, nx)[::spatial_stride, ::spatial_stride].flatten()
        ax.plot_trisurf(x_s, y_s, z_s, cmap=plt.get_cmap(VELOCITY_CMAP))
        ax.set_xlim((grid.x_grid.x_i, grid.x_grid.x_f))
        ax.set_ylim((grid.y_grid.x_i, grid.y_grid.x_f))
        ax.set_zlim((min_z, max_z))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("du/dt")

    ax.view_init(25, 45, 0)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, frames=t_points,
                                   interval=animation_interval, blit=False,
                                   repeat_delay=REPEAT_DELAY)

    if _is_mp4(file_name):
        extra = output_params if output_params is not None else FFMPEG_FAST
        anim.save(file_name, writer="ffmpeg", fps=fps, extra_args=extra)
    else:
        anim.save(file_name, writer="pillow", fps=fps)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Fast imshow backend
# ---------------------------------------------------------------------------

def gen_anim_fast(data: np.ndarray, grid: Grid_2D, file_name: str,
                  z_label: str = "u", duration: float = DURATION,
                  stride: int = 1, spatial_stride: int = 1,
                  output_params: list | None = None) -> None:
    """Fast 2-D heatmap animation using imshow + direct imageio frame writing.

    Parameters
    ----------
    data : ndarray of shape (t_points, Ny*Nx)
    grid : Grid_2D
    file_name : str          — output path (.gif or .mp4)
    z_label : str            — colorbar label
    duration : float         — total animation length in seconds
    stride : int             — render every nth time step
    spatial_stride : int     — sample every nth grid point in x and y
    output_params : list     — ffmpeg output flags (mp4 only); None → ultrafast
    """
    frames = data[::stride]
    t_points = len(frames)
    fps = max(1, t_points / duration)

    nx = grid.x_points
    ny = grid.y_points

    global_min = float(np.min(frames))
    global_max = float(np.max(frames))
    x_extent = [grid.x_grid.x_i, grid.x_grid.x_f,
                grid.y_grid.x_i, grid.y_grid.x_f]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(frames[0].reshape(ny, nx)[::spatial_stride, ::spatial_stride],
                   origin="lower", aspect="auto", cmap=FAST_CMAP,
                   vmin=global_min, vmax=global_max, extent=x_extent)
    fig.colorbar(im, ax=ax).set_label(z_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.canvas.draw()

    with imageio.get_writer(file_name, **_imageio_kwargs(file_name, fps, output_params)) as writer:
        for i in range(t_points):
            im.set_data(frames[i].reshape(ny, nx)[::spatial_stride, ::spatial_stride])
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            writer.append_data(buf.reshape(h, w, 4)[..., :3])

    plt.close(fig)


def gen_velocity_anim_fast(velocity_data: np.ndarray, grid: Grid_2D, file_name: str,
                            duration: float = DURATION, stride: int = 1,
                            spatial_stride: int = 1,
                            output_params: list | None = None) -> None:
    """Fast imshow version of gen_velocity_anim. See gen_anim_fast for details."""
    frames = velocity_data[::stride]
    t_points = len(frames)
    fps = max(1, t_points / duration)

    nx = grid.x_points
    ny = grid.y_points

    global_min = float(np.min(frames))
    global_max = float(np.max(frames))
    x_extent = [grid.x_grid.x_i, grid.x_grid.x_f,
                grid.y_grid.x_i, grid.y_grid.x_f]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(frames[0].reshape(ny, nx)[::spatial_stride, ::spatial_stride],
                   origin="lower", aspect="auto", cmap=FAST_VELOCITY_CMAP,
                   vmin=global_min, vmax=global_max, extent=x_extent)
    fig.colorbar(im, ax=ax).set_label("du/dt")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.canvas.draw()

    with imageio.get_writer(file_name, **_imageio_kwargs(file_name, fps, output_params)) as writer:
        for i in range(t_points):
            im.set_data(frames[i].reshape(ny, nx)[::spatial_stride, ::spatial_stride])
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            writer.append_data(buf.reshape(h, w, 4)[..., :3])

    plt.close(fig)
