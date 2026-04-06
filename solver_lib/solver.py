import numpy as np
from sympy.utilities.lambdify import lambdify
from typing import Union, Callable
import sympy as sp
from grid import Grid_2D, VelocityGrid
from boundary_conditions import BoundaryConditions, DirichletMask
from differential_equation import DifferentialEquation
from animate import (gen_anim, gen_velocity_anim,
                     gen_anim_fast, gen_velocity_anim_fast)

# Type alias for initial conditions
InitialConditionType = Union[sp.Expr, Callable[[np.ndarray, np.ndarray], np.ndarray], int, float]

class Solver:
    def __init__(self,
                 equation: DifferentialEquation,
                 grid: Grid_2D,
                 boundary_conditions: BoundaryConditions,
                 t_i: float,
                 t_f: float,
                 t_points: int,
                 initial_condition: InitialConditionType,
                 initial_velocity: InitialConditionType = None):
        self.equation = equation
        self.grid = grid
        self.bc = boundary_conditions

        self.u_symbol = equation.u_symbol
        self.x_symbol = equation.x_symbol
        self.y_symbol = equation.y_symbol
        self.t_symbol = equation.t_symbol

        self.time_derivative_order = equation.time_derivative_order

        # Time discretization
        self.t_array = np.linspace(t_i, t_f, t_points)
        self.t_delta = (t_f - t_i) / (t_points - 1)
        self.t_points = t_points
        self.current_time = t_i
        self.current_step = 0

        self.solution_history = np.zeros((t_points, grid.x_points * grid.y_points))

        # PDE term storage
        self.derivative_matrices = {}   # (dx_order, dy_order) -> matrix
        self.coefficient_functions = {} # (dx_order, dy_order) -> lambdified fn
        self.source_function = None     # lambdified source term, or None

        # Initialize components
        self.init_derivative_matrices(equation, grid)
        self.init_grid(grid, initial_condition, t_i)
        self.init_velocity(grid, initial_velocity, t_points)

    def init_derivative_matrices(self, equation: DifferentialEquation, grid: Grid_2D) -> None:
        term_dict = equation.extract_spatial_terms()

        for key, coeff_expr in term_dict.items():
            if key == 'source':
                self.source_function = lambdify(
                    [self.x_symbol, self.y_symbol, self.t_symbol], coeff_expr, modules='numpy')
                continue

            dx_order, dy_order = key
            if dx_order + dy_order > 2:
                raise ValueError(
                    f"Term order ({dx_order},{dy_order}) exceeds 2; "
                    "only up to 2nd-order spatial derivatives are supported.")

            self.coefficient_functions[key] = lambdify(
                [self.x_symbol, self.y_symbol, self.t_symbol], coeff_expr, modules='numpy')

            if dx_order == 0 and dy_order == 0:
                pass  # u term — coefficient only, no matrix
            elif dy_order == 0:
                self.derivative_matrices[key] = grid.derivative_matrix(
                    order=dx_order, direction='x')
            elif dx_order == 0:
                self.derivative_matrices[key] = grid.derivative_matrix(
                    order=dy_order, direction='y')
            else:  # (1, 1): u_xy = Dy @ Dx
                Dx = grid.derivative_matrix(order=1, direction='x')
                Dy = grid.derivative_matrix(order=1, direction='y')
                self.derivative_matrices[key] = Dy @ Dx

    def init_grid(self, grid: Grid_2D, initial_condition: InitialConditionType, t_i: float) -> None:
        if callable(initial_condition):
            grid.values = initial_condition(grid.xv, grid.yv).flatten()
        else:
            grid.initialize_values(initial_condition, self.x_symbol, self.y_symbol)

        self.initial_values = grid.values.copy()
        self.t_i = t_i
        self.solution_history[0] = grid.values.copy()

    def init_velocity(self, grid: Grid_2D, initial_velocity: InitialConditionType,
                      t_points: int) -> None:
        if self.time_derivative_order == 2:
            if initial_velocity is None:
                raise ValueError("initial_velocity required for hyperbolic PDEs.")

            if callable(initial_velocity):
                vel_array = initial_velocity(grid.xv, grid.yv).flatten()
            elif isinstance(initial_velocity, (int, float)):
                vel_array = np.full(grid.x_points * grid.y_points, float(initial_velocity))
            else:
                vel_func = lambdify([self.x_symbol, self.y_symbol], initial_velocity, modules='numpy')
                vel_array = vel_func(grid.xv, grid.yv).flatten()

            self.initial_velocity = vel_array

            if isinstance(grid, VelocityGrid):
                # Newmark path: store on grid and pre-allocate velocity_history
                grid.velocity[:] = vel_array
                self.velocity_history = np.zeros((t_points, grid.x_points * grid.y_points))
                self.velocity_history[0] = vel_array.copy()
            else:
                # Leapfrog path: initial_velocity stored on solver; no history yet
                self.velocity_history = None

        elif self.time_derivative_order == 1:
            self.initial_velocity = None
            self.velocity_history = None

        else:
            raise ValueError(f"Unsupported time_derivative_order={self.time_derivative_order}.")

    def compute_dudt(self, u_values: np.ndarray, time: float) -> np.ndarray:
        dudt = np.zeros_like(u_values)

        for key, coeff_func in self.coefficient_functions.items():
            coeff_values = coeff_func(self.grid.xv, self.grid.yv, time)
            coeff = coeff_values if np.isscalar(coeff_values) else coeff_values.flatten()

            if key == (0, 0):
                dudt += coeff * u_values
            else:
                dudt += coeff * (self.derivative_matrices[key] @ u_values)

        if self.source_function is not None:
            source_values = self.source_function(self.grid.xv, self.grid.yv, time)
            dudt += source_values if np.isscalar(source_values) else source_values.flatten()

        return dudt

    def reset(self) -> None:
        self.grid.values = self.initial_values.copy()

        if self.time_derivative_order == 2 and isinstance(self.grid, VelocityGrid):
            self.grid.velocity[:] = self.initial_velocity
            self.velocity_history = np.zeros((self.t_points, self.grid.x_points * self.grid.y_points))
            self.velocity_history[0] = self.initial_velocity.copy()
        elif self.time_derivative_order == 2:
            self.velocity_history = None  # leapfrog path; will be reset on next solve_leapfrog call

        self.current_time = self.t_i
        self.current_step = 0
        self.solution_history = np.zeros((self.t_points, self.grid.x_points * self.grid.y_points))
        self.solution_history[0] = self.initial_values.copy()

    def solve_euler(self) -> np.ndarray:
        if self.time_derivative_order != 1:
            raise ValueError("solve_euler() only supports parabolic PDEs. Use solve_newmark() for hyperbolic.")

        for step in range(1, self.t_points):
            self.current_step = step
            self.current_time = self.t_array[step - 1]

            self.grid.values += self.t_delta * self.compute_dudt(self.grid.values, self.current_time)
            self.grid.apply_boundary_conditions(self.bc, time=self.t_array[step])
            self.solution_history[step] = self.grid.values.copy()

        self.current_time = self.t_array[-1]
        return self.solution_history

    def solve_rk4(self) -> np.ndarray:
        if self.time_derivative_order != 1:
            raise ValueError("solve_rk4() only supports parabolic PDEs. Use solve_newmark() for hyperbolic.")

        for step in range(1, self.t_points):
            self.current_step = step
            t_n = self.t_array[step - 1]
            u_n = self.grid.values.copy()

            k1 = self.compute_dudt(u_n, t_n)

            self.grid.values = u_n + 0.5 * self.t_delta * k1
            if self.bc.is_time_dependent:
                self.grid.apply_boundary_conditions(self.bc, time=t_n + 0.5 * self.t_delta)
            k2 = self.compute_dudt(self.grid.values, t_n + 0.5 * self.t_delta)

            self.grid.values = u_n + 0.5 * self.t_delta * k2
            if self.bc.is_time_dependent:
                self.grid.apply_boundary_conditions(self.bc, time=t_n + 0.5 * self.t_delta)
            k3 = self.compute_dudt(self.grid.values, t_n + 0.5 * self.t_delta)

            self.grid.values = u_n + self.t_delta * k3
            if self.bc.is_time_dependent:
                self.grid.apply_boundary_conditions(self.bc, time=t_n + self.t_delta)
            k4 = self.compute_dudt(self.grid.values, t_n + self.t_delta)

            self.grid.values = u_n + (self.t_delta / 6) * (k1 + 2*k2 + 2*k3 + k4)
            self.grid.apply_boundary_conditions(self.bc, time=self.t_array[step])
            self.solution_history[step] = self.grid.values.copy()

        self.current_time = self.t_array[-1]
        return self.solution_history

    def solve_newmark(self, beta: float = 0.25, gamma: float = 0.5) -> np.ndarray:
        if self.time_derivative_order != 2:
            raise ValueError("solve_newmark() only supports hyperbolic PDEs. Use solve_euler() or solve_rk4() for parabolic.")
        if not isinstance(self.grid, VelocityGrid):
            raise TypeError("solve_newmark() requires a VelocityGrid.")
        if not (0 <= beta <= 0.5):
            raise ValueError(f"Beta parameter {beta} outside stable range [0, 0.5]")
        if not (0 <= gamma <= 1):
            raise ValueError(f"Gamma parameter {gamma} outside valid range [0, 1]")

        u_n = self.grid.values.copy()
        v_n = self.grid.velocity.copy()

        for step in range(1, self.t_points):
            self.current_step = step
            t_n = self.t_array[step - 1]
            t_np1 = self.t_array[step]

            a_n = self.compute_dudt(u_n, t_n)

            u_star = u_n + self.t_delta * v_n + (self.t_delta ** 2 / 2) * (1 - 2 * beta) * a_n
            v_star = v_n + self.t_delta * (1 - gamma) * a_n

            self.grid.values = u_star.copy()
            self.grid.velocity[:] = v_star
            self.grid.apply_boundary_conditions(self.bc, time=t_np1)
            u_star = self.grid.values.copy()
            v_star = self.grid.velocity.copy()

            a_star = self.compute_dudt(u_star, t_np1)

            u_np1 = u_star + self.t_delta ** 2 * beta * a_star
            v_np1 = v_star + self.t_delta * gamma * a_star

            self.grid.values = u_np1.copy()
            self.grid.velocity[:] = v_np1
            self.grid.apply_boundary_conditions(self.bc, time=t_np1)
            u_np1 = self.grid.values.copy()
            v_np1 = self.grid.velocity.copy()

            self.solution_history[step] = u_np1.copy()
            self.velocity_history[step] = v_np1.copy()

            u_n = u_np1
            v_n = v_np1

        self.grid.values = u_n
        self.grid.velocity[:] = v_n
        self.current_time = self.t_array[-1]
        return self.solution_history

    def get_solution_at_time(self, time_index: int) -> np.ndarray:
        return self.solution_history[time_index]

    def get_solution_2d(self, time_index: int) -> np.ndarray:
        return self.solution_history[time_index].reshape((self.grid.y_points, self.grid.x_points))

    def get_velocity_at_time(self, time_index: int) -> np.ndarray:
        if self.velocity_history is None:
            raise AttributeError(
                "Velocity history not available. solve_leapfrog() does not track velocity "
                "explicitly. Use solve_newmark() if velocity data is required.")
        return self.velocity_history[time_index]

    def get_velocity_2d(self, time_index: int) -> np.ndarray:
        if self.velocity_history is None:
            raise AttributeError(
                "Velocity history not available. solve_leapfrog() does not track velocity "
                "explicitly. Use solve_newmark() if velocity data is required.")
        return self.velocity_history[time_index].reshape((self.grid.y_points, self.grid.x_points))

    def animate(self, file_name: str, z_label: str = "u", duration: float = 5.0,
                output_type: str = "3D", stride: int = 1, spatial_stride: int = 1,
                output_params: list = None) -> None:
        """Render solution history to an animation file.

        Parameters
        ----------
        file_name : str        — output path (.gif or .mp4)
        z_label : str          — z-axis / colorbar label
        duration : float       — animation length in seconds
        output_type : str      — ``"3D"`` or ``"2D"``
        stride : int           — render every nth time step
        spatial_stride : int   — sample every nth grid point in x and y
        output_params : list   — ffmpeg flags for .mp4;
                                 None → ["-preset", "ultrafast", "-crf", "23"]
        """
        if output_type == "2D":
            gen_anim_fast(self.solution_history, self.grid, file_name, z_label,
                          duration, stride, spatial_stride, output_params)
        else:
            gen_anim(self.solution_history, self.grid, file_name, z_label,
                     duration, stride, spatial_stride, output_params)

    def animate_velocity(self, file_name: str, duration: float = 5.0,
                         output_type: str = "3D", stride: int = 1,
                         spatial_stride: int = 1,
                         output_params: list = None) -> None:
        """Render velocity history to an animation file.

        Parameters
        ----------
        file_name : str        — output path (.gif or .mp4)
        duration : float       — animation length in seconds
        output_type : str      — ``"3D"`` or ``"2D"``
        stride : int           — render every nth time step
        spatial_stride : int   — sample every nth grid point in x and y
        output_params : list   — ffmpeg flags for .mp4
        """
        if self.velocity_history is None:
            raise AttributeError(
                "Velocity history not available. solve_leapfrog() does not track velocity "
                "explicitly. Use solve_newmark() if velocity data is required.")
        if output_type == "2D":
            gen_velocity_anim_fast(self.velocity_history, self.grid, file_name,
                                   duration, stride, spatial_stride, output_params)
        else:
            gen_velocity_anim(self.velocity_history, self.grid, file_name,
                              duration, stride, spatial_stride, output_params)

    # ------------------------------------------------------------------
    # Leapfrog helpers and solver
    # ------------------------------------------------------------------

    def _extract_c_sq(self, grid: Grid_2D) -> np.ndarray:
        """Evaluate c²(x,y) from the (2,0) spatial term coefficient on the grid."""
        coeff_fn = self.coefficient_functions.get((2, 0))
        if coeff_fn is None:
            raise ValueError(
                "solve_leapfrog() requires a u_xx term in the equation "
                "(no (2,0) spatial term found).")
        c_sq = coeff_fn(grid.xv, grid.yv, 0.0)
        if np.isscalar(c_sq):
            return np.full((grid.y_points, grid.x_points), float(c_sq))
        return np.asarray(c_sq, dtype=float).reshape(grid.y_points, grid.x_points)

    def _validate_c_sq_time_independence(self) -> None:
        """Raise ValueError if any Laplacian coefficient contains the time symbol."""
        term_dict = self.equation.extract_spatial_terms()
        for key in [(2, 0), (0, 2)]:
            expr = term_dict.get(key, sp.Integer(0))
            if expr.has(self.t_symbol):
                raise ValueError(
                    f"solve_leapfrog() requires c² to be time-independent, but the "
                    f"coefficient for term {key} contains the time variable.")

    def solve_leapfrog(self, gamma: float = 0.25) -> np.ndarray:
        """
        Solve the wave equation using leapfrog time-stepping with Kreiss-Petersson
        embedded boundary conditions.

        Ghost points (grid points outside the domain that border interior points)
        are eliminated algebraically at preprocessing time.  The embedded BC is
        baked into a sparse modified Laplacian matrix A and a boundary forcing
        vector b(t), so no per-step BC enforcement is needed.

        Leapfrog update:
            u^{n+1} = 2 u^n - u^{n-1} + dt² [A u^n + b(t_n) + F(t_n)]

        Initialization (2nd-order start):
            u^{-1} = u^0 - dt * u1

        Parameters
        ----------
        gamma : float
            K-P stabilization parameter. Must be >= 0.25 to guarantee that the
            CFL condition k/h < 1/√2 is preserved regardless of small-cell fraction.

        Returns
        -------
        solution_history : ndarray of shape (t_points, Ny*Nx)
            Full-grid displacement at each time step. Points outside the domain
            (ghost and far-exterior) are set to 0.
        """
        if self.time_derivative_order != 2:
            raise ValueError(
                "solve_leapfrog() only supports hyperbolic PDEs (time_derivative_order=2). "
                "Use solve_euler() or solve_rk4() for parabolic PDEs.")
        if not isinstance(self.bc, DirichletMask):
            raise TypeError(
                "solve_leapfrog() requires a DirichletMask boundary condition. "
                "Use solve_newmark() for axis-aligned BoundaryConditions.")
        if gamma < 0.25:
            raise ValueError(
                f"gamma={gamma} < 0.25 violates the K-P CFL-preservation condition. "
                "Use gamma >= 0.25.")
        self._validate_c_sq_time_independence()

        # Preprocess (result is cached on DirichletMask after the first call)
        c_sq_2d = self._extract_c_sq(self.grid)
        A, b_func, _, interior_to_full, _ = self.bc.preprocess_kp(self.grid, c_sq_2d, gamma)
        N_int = len(interior_to_full)

        # Initial conditions at interior dofs only
        u_curr = self.initial_values[interior_to_full].copy()   # u^0
        u1     = self.initial_velocity[interior_to_full]         # du/dt at t=0
        u_prev = u_curr - self.t_delta * u1                      # u^{-1}

        def get_F(t):
            if self.source_function is None:
                return np.zeros(N_int)
            F_full = self.source_function(self.grid.xv, self.grid.yv, t)
            if np.isscalar(F_full):
                return np.full(N_int, float(F_full))
            return F_full.flatten()[interior_to_full]

        # Allocate history: full grid arrays, exterior stays 0
        self.solution_history = np.zeros((self.t_points,
                                          self.grid.x_points * self.grid.y_points))
        self.solution_history[0, interior_to_full] = u_curr

        dt_sq = self.t_delta ** 2
        for step in range(1, self.t_points):
            t_n = self.t_i + (step - 1) * self.t_delta
            rhs = A @ u_curr + b_func(t_n) + get_F(t_n)
            u_next = 2.0 * u_curr - u_prev + dt_sq * rhs
            
            self.solution_history[step, interior_to_full] = u_next
            u_prev = u_curr
            u_curr = u_next

        self.velocity_history = None  # leapfrog does not track velocity
        self.grid.values[interior_to_full] = u_curr
        self.current_time = self.t_array[-1]
        return self.solution_history
