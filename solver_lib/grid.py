import numpy as np
from sympy.utilities.lambdify import lambdify
from typing import Callable, Any
from boundary_conditions import BoundaryConditions, DirichletMask

# Import coefficient functions
from fd_coeffs import (
    neumann_boundary_forward,
    neumann_boundary_backward,
    calculate_fd_coefficients
)

class Grid_1D:
    def __init__(self, points: int, x_i: float = 0, x_f: float = 1):
        self.points = points
        self.x_i = x_i
        self.x_f = x_f
        self.delta = (x_f - x_i) / (points - 1)
        self.x = np.linspace(x_i, x_f, points)
        self.values = np.zeros(points)

    def initialize_values(self, func: Any, x_symbol: Any) -> None:
        func_vectorized = np.vectorize(lambdify([x_symbol], func))
        self.values = func_vectorized(self.x)

    def set_boundary_dirichlet(self, lower_val: float = None, upper_val: float = None) -> None: # type: ignore
        if lower_val is not None:
            self.values[0] = lower_val
        if upper_val is not None:
            self.values[-1] = upper_val

    def set_boundary_neumann(self, lower_deriv: float = None, upper_deriv: float = None, # type: ignore
                           accuracy_order: int = 1) -> None:
        # Validate grid size
        min_points = accuracy_order + 2
        if self.points < min_points:
            raise ValueError(
                f"Grid needs at least {min_points} points for accuracy_order={accuracy_order}, "
                f"but has {self.points} points"
            )

        # Apply left boundary (x_i) if specified
        if lower_deriv is not None:
            neumann_scale, interior_coeffs = neumann_boundary_forward(self.delta, accuracy_order)
            interior_vals = self.values[1:1+accuracy_order]
            self.values[0] = neumann_scale * lower_deriv + np.dot(interior_coeffs, interior_vals)

        # Apply right boundary (x_f) if specified
        if upper_deriv is not None:
            neumann_scale, interior_coeffs = neumann_boundary_backward(self.delta, accuracy_order)
            # Extract interior values from nearest to farthest (reverse order)
            interior_vals = self.values[-accuracy_order-1:-1][::-1]
            self.values[-1] = neumann_scale * upper_deriv + np.dot(interior_coeffs, interior_vals)

    def derivative_matrix(self, order: int = 2, accuracy_order: int = 2,
                         strategy: str = 'custom_stencil') -> np.ndarray:
        # Calculate number of points needed for stencil
        num_points = order + accuracy_order

        mat = np.zeros((self.points, self.points))

        if strategy == 'forward_central_backward':
            # Forward stencil: [0, 1, 2, ..., num_points-1]
            forward_stencil = np.arange(num_points)
            forward_coeffs, _ = calculate_fd_coefficients(forward_stencil, order)

            # Central stencil: symmetric around 0
            central_num_points = 2 * ((order + accuracy_order - 1) // 2) + 1
            half = central_num_points // 2
            central_stencil = np.arange(-half, half + 1)
            central_coeffs, _ = calculate_fd_coefficients(central_stencil, order)

            # Backward stencil: [0, -1, -2, ..., -(num_points-1)]
            backward_stencil = -np.arange(num_points)
            backward_coeffs, _ = calculate_fd_coefficients(backward_stencil, order)

            # Determine transition points
            forward_region_end = half
            backward_region_start = self.points - half

            # forward
            for i in range(min(forward_region_end, self.points)):
                mat[i, i:i+num_points] = forward_coeffs / (self.delta ** order)

            # central
            for i in range(forward_region_end, min(backward_region_start, self.points)):
                start_col = i - half
                end_col = i + half + 1
                mat[i, start_col:end_col] = central_coeffs / (self.delta ** order)

            # backward
            for i in range(max(backward_region_start, 0), self.points):
                mat[i, i-num_points+1:i+1] = backward_coeffs / (self.delta ** order)
        else:  # strategy == 'custom_stencil' (default)
            for i in range(self.points):
                # Determine the stencil centered around point i
                # We want num_points total points in the stencil

                # Ideally, center the stencil around i
                half = num_points // 2
                stencil_start = i - half
                stencil_end = i + (num_points - half)

                # Adjust if we're near boundaries
                if stencil_start < 0:
                    # Too close to left boundary, shift right
                    stencil_start = 0
                    stencil_end = num_points
                elif stencil_end > self.points:
                    # Too close to right boundary, shift left
                    stencil_end = self.points
                    stencil_start = self.points - num_points

                # Create the stencil relative to point i
                # Stencil points are at positions: stencil_start, stencil_start+1, ..., stencil_end-1
                # Relative to i, these are at: stencil_start-i, stencil_start+1-i, ..., stencil_end-1-i
                stencil = np.arange(stencil_start - i, stencil_end - i)

                # Calculate coefficients for this stencil
                coeffs, _ = calculate_fd_coefficients(stencil, order)

                # Place coefficients in the matrix
                mat[i, stencil_start:stencil_end] = coeffs / (self.delta ** order)

        return mat

    def laplacian_matrix(self, accuracy_order: int = 2, strategy: str = 'custom_stencil') -> np.ndarray:
        return self.derivative_matrix(order=2, accuracy_order=accuracy_order, strategy=strategy)

def _evaluate_bc_func(func, spatial_coords, time):
    """Helper to evaluate BC function with or without time parameter."""
    if time is None:
        # Time-independent evaluation
        return func(spatial_coords)
    else:
        # Try time-dependent evaluation first
        try:
            return func(spatial_coords, time)
        except TypeError:
            # Function doesn't accept time parameter, fall back to time-independent
            return func(spatial_coords)

class Grid_2D:
    def __init__(self, x_points: int, y_points: int,
                 x_i: float = 0, x_f: float = 1, y_i: float = 0, y_f: float = 1,
                 accuracy_order: int = 2, strategy: str = 'custom_stencil'):
        self.x_grid = Grid_1D(x_points, x_i, x_f)
        self.y_grid = Grid_1D(y_points, y_i, y_f)

        self.x_points = x_points
        self.y_points = y_points

        self.accuracy_order = accuracy_order
        self.strategy = strategy

        # Meshgrid for coordinate points
        self.xv, self.yv = np.meshgrid(self.x_grid.x, self.y_grid.x)

        # Flattened values array
        self.values = np.zeros(x_points * y_points)

    def initialize_values(self, func: Any, x_symbol: Any, y_symbol: Any) -> None:
        func_vectorized = np.vectorize(lambdify([x_symbol, y_symbol], func))
        values_2d = func_vectorized(self.xv, self.yv)
        self.values = values_2d.flatten()

    def set_boundary_dirichlet(self, x_0_func: Callable = None, x_L_func: Callable = None, # type: ignore
                              y_0_func: Callable = None, y_L_func: Callable = None,
                              time: float = None) -> None:
        # Bottom and top boundaries (only if specified)
        if y_0_func is not None:
            self.values[0:self.x_points] = _evaluate_bc_func(y_0_func, self.x_grid.x, time)
        if y_L_func is not None:
            self.values[-self.x_points:] = _evaluate_bc_func(y_L_func, self.x_grid.x, time)

        # Left boundary (x = x_i)
        if x_0_func is not None:
            index = 0
            for x_val in _evaluate_bc_func(x_0_func, self.y_grid.x, time):
                self.values[index] = x_val
                index += self.x_points

        # Right boundary (x = x_f)
        if x_L_func is not None:
            index = self.x_points - 1
            for x_val in _evaluate_bc_func(x_L_func, self.y_grid.x, time):
                self.values[index] = x_val
                index += self.x_points

    def set_boundary_neumann(self, x_0_deriv: Callable = None, x_L_deriv: Callable = None,
                            y_0_deriv: Callable = None, y_L_deriv: Callable = None,
                            corner_mode: str = 'average',
                            time: float = None) -> None:
        # corner_mode: How to handle corners ('average', 'x_priority', 'y_priority')
        # time: Current time value. If None, only spatial coordinates are passed to functions.

        accuracy_order = self.accuracy_order

        # Validate grid sizes
        min_x_points = accuracy_order + 2
        min_y_points = accuracy_order + 2
        if self.x_points < min_x_points or self.y_points < min_y_points:
            raise ValueError(
                f"Grid needs at least {min_x_points}x{min_y_points} points for accuracy_order={accuracy_order}, "
                f"but has {self.x_points}x{self.y_points} points"
            )

        # Validate corner_mode
        if corner_mode not in ['average', 'x_priority', 'y_priority']:
            raise ValueError(f"Invalid corner_mode '{corner_mode}'. Must be 'average', 'x_priority', or 'y_priority'")

        # Store corner indices for later handling
        corner_indices = [
            0,  # Bottom-left (x_i, y_i)
            self.x_points - 1,  # Bottom-right (x_f, y_i)
            (self.y_points - 1) * self.x_points,  # Top-left (x_i, y_f)
            (self.y_points - 1) * self.x_points + self.x_points - 1  # Top-right (x_f, y_f)
        ]

        # Store corner values before modification (for averaging)
        corner_values_from_y = {}
        corner_values_from_x = {}

        # Determine application order based on corner_mode
        if corner_mode == 'x_priority':
            # Apply y boundaries first, then x (x will override)
            apply_order = ['y', 'x']
        elif corner_mode == 'y_priority':
            # Apply x boundaries first, then y (y will override)
            apply_order = ['x', 'y']
        else:  # 'average'
            # Apply in y, x order, then average corners
            apply_order = ['y', 'x']

        for direction in apply_order:
            if direction == 'y':
                # Bottom boundary (y = y_i) - if y_0_deriv is not None
                if y_0_deriv is not None:
                    neumann_scale, interior_coeffs = neumann_boundary_forward(self.y_grid.delta, accuracy_order)
                    y_0_deriv_vals = _evaluate_bc_func(y_0_deriv, self.x_grid.x, time)

                    for i in range(self.x_points):
                        # Extract interior values in y-direction (rows 1 to accuracy_order)
                        # These correspond to f[1], f[2], ..., f[accuracy_order]
                        interior_vals = np.array([self.values[i + k * self.x_points]
                                                for k in range(1, accuracy_order + 1)])
                        new_val = neumann_scale * y_0_deriv_vals[i] + np.dot(interior_coeffs, interior_vals)

                        if corner_mode == 'average' and i in [0, self.x_points - 1]:
                            corner_values_from_y[i] = new_val
                        else:
                            self.values[i] = new_val

                # Top boundary (y = y_f) - if y_L_deriv is not None
                if y_L_deriv is not None:
                    neumann_scale, interior_coeffs = neumann_boundary_backward(self.y_grid.delta, accuracy_order)
                    y_L_deriv_vals = _evaluate_bc_func(y_L_deriv, self.x_grid.x, time)

                    for i in range(self.x_points):
                        # Extract interior values: f[N-1], f[N-2], ..., f[N-accuracy_order]
                        # (nearest to farthest from boundary)
                        interior_vals = np.array([self.values[i + (self.y_points - 1 - k) * self.x_points]
                                                for k in range(1, accuracy_order + 1)])
                        top_idx = i + (self.y_points - 1) * self.x_points
                        new_val = neumann_scale * y_L_deriv_vals[i] + np.dot(interior_coeffs, interior_vals)

                        if corner_mode == 'average' and top_idx in corner_indices:
                            corner_values_from_y[top_idx] = new_val
                        else:
                            self.values[top_idx] = new_val

            else:  # direction == 'x'
                # Left boundary (x = x_i) - if x_0_deriv is not None
                if x_0_deriv is not None:
                    neumann_scale, interior_coeffs = neumann_boundary_forward(self.x_grid.delta, accuracy_order)
                    x_0_deriv_vals = _evaluate_bc_func(x_0_deriv, self.y_grid.x, time)

                    for j in range(self.y_points):
                        # Extract interior values in x-direction: f[1], f[2], ..., f[accuracy_order]
                        interior_vals = np.array([self.values[j * self.x_points + k]
                                                for k in range(1, accuracy_order + 1)])
                        left_idx = j * self.x_points
                        new_val = neumann_scale * x_0_deriv_vals[j] + np.dot(interior_coeffs, interior_vals)

                        if corner_mode == 'average' and left_idx in corner_indices:
                            corner_values_from_x[left_idx] = new_val
                        else:
                            self.values[left_idx] = new_val

                # Right boundary (x = x_f) - if x_L_deriv is not None
                if x_L_deriv is not None:
                    neumann_scale, interior_coeffs = neumann_boundary_backward(self.x_grid.delta, accuracy_order)
                    x_L_deriv_vals = _evaluate_bc_func(x_L_deriv, self.y_grid.x, time)

                    for j in range(self.y_points):
                        # Extract interior values: f[N-1], f[N-2], ..., f[N-accuracy_order]
                        # (nearest to farthest from boundary)
                        interior_vals = np.array([self.values[j * self.x_points + (self.x_points - 1 - k)]
                                                for k in range(1, accuracy_order + 1)])
                        right_idx = j * self.x_points + (self.x_points - 1)
                        new_val = neumann_scale * x_L_deriv_vals[j] + np.dot(interior_coeffs, interior_vals)

                        if corner_mode == 'average' and right_idx in corner_indices:
                            corner_values_from_x[right_idx] = new_val
                        else:
                            self.values[right_idx] = new_val

        # Handle corners in averaging mode
        if corner_mode == 'average':
            for corner_idx in corner_indices:
                if corner_idx in corner_values_from_x and corner_idx in corner_values_from_y:
                    # Both boundaries affected this corner - average them
                    self.values[corner_idx] = (corner_values_from_x[corner_idx] + corner_values_from_y[corner_idx]) / 2.0
                elif corner_idx in corner_values_from_x:
                    # Only x boundary affected this corner
                    self.values[corner_idx] = corner_values_from_x[corner_idx]
                elif corner_idx in corner_values_from_y:
                    # Only y boundary affected this corner
                    self.values[corner_idx] = corner_values_from_y[corner_idx]

    def apply_boundary_conditions(self, bc: BoundaryConditions | DirichletMask,
                                  corner_mode: str = 'average', time: float = None,
                                  pseudo_boundary_n: int = 1) -> None:
        # corner_mode: How to handle corners ('average', 'x_priority', 'y_priority')
        # time: Current time value. If None, only spatial coordinates are passed to functions.
        # pseudo_boundary_n: stencil half-width used to detect pseudo-boundary points
        #   for DirichletMask BCs (passed through to apply_dirichlet_mask).

        # handle DirichletMask case separately
        if isinstance(bc, DirichletMask):
            self.apply_dirichlet_mask(bc, n=pseudo_boundary_n)
            return

        values_2d = self.values.reshape((self.y_points, self.x_points))

        # Apply Dirichlet boundaries
        dirichlet_bcs = bc.dirichlet_boundaries
        for boundary_name, func in dirichlet_bcs.items():
            if boundary_name == 'y_0':
                values_2d[0, :] = _evaluate_bc_func(func, self.x_grid.x, time)
            elif boundary_name == 'y_L':
                values_2d[-1, :] = _evaluate_bc_func(func, self.x_grid.x, time)
            elif boundary_name == 'x_0':
                values_2d[:, 0] = _evaluate_bc_func(func, self.y_grid.x, time)
            elif boundary_name == 'x_L':
                values_2d[:, -1] = _evaluate_bc_func(func, self.y_grid.x, time)

        self.values = values_2d.flatten()

        # Apply Neumann boundaries
        neumann_bcs = bc.neumann_boundaries
        if neumann_bcs:
            neumann_args = {
                'x_0_deriv': neumann_bcs.get('x_0'),
                'x_L_deriv': neumann_bcs.get('x_L'),
                'y_0_deriv': neumann_bcs.get('y_0'),
                'y_L_deriv': neumann_bcs.get('y_L'),
                'corner_mode': corner_mode,
                'time': time
            }
            self.set_boundary_neumann(**neumann_args)

    def derivative_matrix(self, order: int = 2, direction: str = 'xy') -> np.ndarray:
        if direction == 'xy':
            if order != 2:
                raise ValueError("Laplacian only defined for second derivatives (order=2)")
            A_x = self.x_grid.derivative_matrix(order=2, accuracy_order=self.accuracy_order, strategy=self.strategy)
            A_y = self.y_grid.derivative_matrix(order=2, accuracy_order=self.accuracy_order, strategy=self.strategy)
            return np.kron(A_y, np.eye(self.x_points)) + np.kron(np.eye(self.y_points), A_x)
        elif direction == 'x':
            D_x = self.x_grid.derivative_matrix(order=order, accuracy_order=self.accuracy_order, strategy=self.strategy)
            return np.kron(np.eye(self.y_points), D_x)
        elif direction == 'y':
            D_y = self.y_grid.derivative_matrix(order=order, accuracy_order=self.accuracy_order, strategy=self.strategy)
            return np.kron(D_y, np.eye(self.x_points))
        else:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'xy', 'x', or 'y'")

    def laplacian_matrix(self) -> np.ndarray:
        return self.derivative_matrix(order=2, direction='xy')

    def apply_dirichlet_mask(self, bc: DirichletMask, n: int = 1) -> None:
        """Enforce bc.dirichlet_value at all masked (exterior) points.

        This is called by apply_boundary_conditions when the BC is a DirichletMask.
        It provides the per-step enforcement needed by the heat equation (Euler/RK4)
        path.  The leapfrog path never calls this method — BCs are baked into the
        precomputed matrix A via preprocess_kp().

        n : stencil half-width — used to pre-warm the pseudo-boundary cache so
            the first call to compute_pseudo_boundary_derivative is not cold.
        """
        mask = bc.get_mask(self)
        self.values[mask.flatten()] = bc.dirichlet_value
        bc.get_pseudo_boundary_points(self, n)  # pre-warm cache

class VelocityGrid(Grid_2D):
    """Grid_2D subclass that owns a velocity field for second-order (hyperbolic) PDEs.

    The velocity field `self.velocity` mirrors the layout of `self.values` (flat,
    row-major).  When a DirichletMask boundary condition is applied, both the
    displacement field (`self.values`) and the velocity field (`self.velocity`)
    are enforced at masked points, using `bc.dirichlet_value` and
    `bc.velocity_value` respectively.

    DEPRECATED for wave equation: Use plain Grid_2D with solve_leapfrog() instead.
    VelocityGrid + solve_newmark() remains supported for the damped wave equation
    and any other scenario that requires explicit velocity-field tracking.
    """

    def __init__(self, x_points: int, y_points: int,
                 x_i: float = 0, x_f: float = 1,
                 y_i: float = 0, y_f: float = 1,
                 accuracy_order: int = 2, strategy: str = 'custom_stencil'):
        super().__init__(x_points, y_points, x_i, x_f, y_i, y_f, accuracy_order, strategy)
        self.velocity = np.zeros(x_points * y_points)

    def apply_dirichlet_mask(self, bc: DirichletMask, n: int = 1) -> None:
        super().apply_dirichlet_mask(bc, n)
        mask = bc.get_mask(self)
        self.velocity[mask.flatten()] = bc.velocity_value
