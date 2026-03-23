from typing import Callable, Optional
import numpy as np
import scipy.sparse as sp
import inspect

from fd_coeffs import (
    calculate_fd_coefficients,
    dirichlet_pseudo_boundary_forward,
    dirichlet_pseudo_boundary_backward,
    lagrange_weights,
    normal_lagrange_weights,
)

class _GridProxy:
    """
    Minimal grid-like object used to evaluate a DirichletMask's mask_function
    at a single arbitrary (x, y) point during binary-search boundary finding.

    The mask_function must depend only on grid.xv and grid.yv (standard for
    geometric masks such as the circular drum).
    """
    def __init__(self, x: float, y: float):
        self.xv = np.array([[x]], dtype=float)
        self.yv = np.array([[y]], dtype=float)
        self.x_points = 1
        self.y_points = 1

class DirichletMask:
    def __init__(self, mask_function: Callable, dirichlet_value: float = 0.0,
                 velocity_value: float = 0.0):
        self.mask_function = mask_function
        self.dirichlet_value = dirichlet_value
        self.velocity_value = velocity_value
        self._cached_mask = None
        self._cached_grid_shape = None
        # Pseudo-boundary cache (used by heat equation path)
        self._pseudo_boundary_cache = None
        self._pseudo_boundary_n = None
        self._pseudo_boundary_grid_shape = None
        # K-P embedded boundary cache (used by solve_leapfrog)
        self._kp_cache = {}

    def get_mask(self, grid):
        # Cache the mask to avoid recomputing it at every time step
        grid_shape = (grid.x_points, grid.y_points)
        if self._cached_mask is None or self._cached_grid_shape != grid_shape:
            self._cached_mask = self.mask_function(grid)
            self._cached_grid_shape = grid_shape
        return self._cached_mask

    # ------------------------------------------------------------------
    # Pseudo-boundary support
    # ------------------------------------------------------------------

    def _find_boundary_distance(self, x_from: float, y_from: float,
                                x_to: float, y_to: float) -> float:
        """
        Binary-search for the physical distance from (x_from, y_from) — an
        interior point — to the mask boundary along the segment toward
        (x_to, y_to) — a masked (exterior) point.

        The mask_function is evaluated via _GridProxy, so it must depend only
        on grid.xv / grid.yv (true for all standard geometric masks).
        """
        lo, hi = 0.0, 1.0   # parametric t along the segment
        for _ in range(50):
            mid = (lo + hi) / 2.0
            x_mid = x_from + mid * (x_to - x_from)
            y_mid = y_from + mid * (y_to - y_from)
            is_outside = bool(self.mask_function(_GridProxy(x_mid, y_mid)).flat[0])
            if is_outside:
                hi = mid
            else:
                lo = mid
        t = (lo + hi) / 2.0
        return t * np.sqrt((x_to - x_from) ** 2 + (y_to - y_from) ** 2)

    def get_pseudo_boundary_points(self, grid, n: int = 1) -> dict:
        """
        Find and cache all interior points that are within *n* grid steps of a
        masked point in any axis-aligned direction.

        These "pseudo-boundary" points are where a standard central-difference
        stencil of half-width *n* would reach outside the domain mask.

        Parameters
        ----------
        grid : Grid_2D
        n    : stencil half-width to check (default 1)

        Returns
        -------
        dict mapping flat_index -> dict with any of the keys:
            'x_pos' : float  physical distance to mask boundary in +x
            'x_neg' : float  physical distance to mask boundary in -x
            'y_pos' : float  physical distance to mask boundary in +y
            'y_neg' : float  physical distance to mask boundary in -y

        Only directions that actually have a masked neighbor within n steps are
        present in the inner dict.  Only points with at least one such direction
        appear in the outer dict.
        """
        grid_shape = (grid.x_points, grid.y_points)
        if (self._pseudo_boundary_cache is not None
                and self._pseudo_boundary_n == n
                and self._pseudo_boundary_grid_shape == grid_shape):
            return self._pseudo_boundary_cache

        mask = self.get_mask(grid)   # shape (y_points, x_points), True = outside domain
        x_coords = grid.x_grid.x    # 1-D array of x positions
        y_coords = grid.y_grid.x    # 1-D array of y positions

        result = {}

        for j in range(grid.y_points):
            for i in range(grid.x_points):
                if mask[j, i]:
                    continue    # skip points that are themselves masked

                info = {}

                # +x direction: look at columns i+1, i+2, ..., i+n
                for k in range(1, n + 1):
                    if i + k >= grid.x_points:
                        break
                    if mask[j, i + k]:
                        info['x_pos'] = self._find_boundary_distance(
                            x_coords[i], y_coords[j],
                            x_coords[i + k], y_coords[j])
                        break

                # -x direction: columns i-1, i-2, ..., i-n
                for k in range(1, n + 1):
                    if i - k < 0:
                        break
                    if mask[j, i - k]:
                        info['x_neg'] = self._find_boundary_distance(
                            x_coords[i], y_coords[j],
                            x_coords[i - k], y_coords[j])
                        break

                # +y direction: rows j+1, j+2, ..., j+n
                for k in range(1, n + 1):
                    if j + k >= grid.y_points:
                        break
                    if mask[j + k, i]:
                        info['y_pos'] = self._find_boundary_distance(
                            x_coords[i], y_coords[j],
                            x_coords[i], y_coords[j + k])
                        break

                # -y direction: rows j-1, j-2, ..., j-n
                for k in range(1, n + 1):
                    if j - k < 0:
                        break
                    if mask[j - k, i]:
                        info['y_neg'] = self._find_boundary_distance(
                            x_coords[i], y_coords[j],
                            x_coords[i], y_coords[j - k])
                        break

                if info:
                    result[j * grid.x_points + i] = info

        self._pseudo_boundary_cache = result
        self._pseudo_boundary_n = n
        self._pseudo_boundary_grid_shape = grid_shape
        return result

    def compute_pseudo_boundary_derivative(self, grid, flat_idx: int,
                                           direction: str,
                                           derivative_order: int,
                                           n: int = 1) -> float:
        """
        Compute the derivative (in *direction*) at a pseudo-boundary point,
        using a stencil that replaces the out-of-domain grid point(s) with the
        known Dirichlet boundary value at the exact geometric boundary.

        The FD stencil is built with interior grid points on the side away from
        the boundary, plus a fractional-step boundary point at the real mask
        edge (found via binary search).  Coefficients come from
        dirichlet_pseudo_boundary_forward / _backward in fd_coeffs.py.

        Parameters
        ----------
        grid             : Grid_2D
        flat_idx         : flat (row-major) index of the pseudo-boundary point
        direction        : 'x' or 'y'
        derivative_order : order of the spatial derivative (1, 2, …)
        n                : stencil half-width (must match the n used when the
                           pseudo-boundary points were found; default 1)

        Returns
        -------
        float : approximate derivative value at that point
        """
        pb_points = self.get_pseudo_boundary_points(grid, n)
        if flat_idx not in pb_points:
            raise ValueError(
                f"flat_idx {flat_idx} is not a pseudo-boundary point for n={n}")

        info = pb_points[flat_idx]
        j, i = divmod(flat_idx, grid.x_points)

        if direction == 'x':
            h = grid.x_grid.delta
            pos_alpha = info.get('x_pos')   # boundary in +x
            neg_alpha = info.get('x_neg')   # boundary in -x

            if pos_alpha is not None and neg_alpha is not None:
                # Both sides masked: only the current point is interior.
                # Stencil: {-neg_alpha/h, 0, pos_alpha/h}
                stencil = np.array([-neg_alpha / h, 0.0, pos_alpha / h])
                coeffs, _ = calculate_fd_coefficients(stencil, derivative_order)
                val = (coeffs[0] * self.dirichlet_value
                       + coeffs[1] * grid.values[flat_idx]
                       + coeffs[2] * self.dirichlet_value)

            elif pos_alpha is not None:
                # Boundary in +x → interior points go left: offsets {-n_avail, ..., 0}
                n_avail = min(n, i)
                col_offsets = np.arange(-n_avail, 1, dtype=float)
                interior_coeffs, boundary_coeff = dirichlet_pseudo_boundary_forward(
                    pos_alpha / h, derivative_order, n_avail)
                interior_vals = np.array([
                    grid.values[j * grid.x_points + (i + int(o))]
                    for o in col_offsets])
                val = np.dot(interior_coeffs, interior_vals) + boundary_coeff * self.dirichlet_value

            else:   # neg_alpha is not None
                # Boundary in -x → interior points go right: offsets {0, ..., n_avail}
                n_avail = min(n, grid.x_points - 1 - i)
                col_offsets = np.arange(0, n_avail + 1, dtype=float)
                interior_coeffs, boundary_coeff = dirichlet_pseudo_boundary_backward(
                    neg_alpha / h, derivative_order, n_avail)
                interior_vals = np.array([
                    grid.values[j * grid.x_points + (i + int(o))]
                    for o in col_offsets])
                val = boundary_coeff * self.dirichlet_value + np.dot(interior_coeffs, interior_vals)

            return val / (h ** derivative_order)

        elif direction == 'y':
            h = grid.y_grid.delta
            pos_alpha = info.get('y_pos')   # boundary in +y (increasing row j)
            neg_alpha = info.get('y_neg')   # boundary in -y (decreasing row j)

            if pos_alpha is not None and neg_alpha is not None:
                stencil = np.array([-neg_alpha / h, 0.0, pos_alpha / h])
                coeffs, _ = calculate_fd_coefficients(stencil, derivative_order)
                val = (coeffs[0] * self.dirichlet_value
                       + coeffs[1] * grid.values[flat_idx]
                       + coeffs[2] * self.dirichlet_value)

            elif pos_alpha is not None:
                # Boundary in +y → interior rows go downward: offsets {-n_avail, ..., 0}
                n_avail = min(n, j)
                row_offsets = np.arange(-n_avail, 1, dtype=float)
                interior_coeffs, boundary_coeff = dirichlet_pseudo_boundary_forward(
                    pos_alpha / h, derivative_order, n_avail)
                interior_vals = np.array([
                    grid.values[(j + int(o)) * grid.x_points + i]
                    for o in row_offsets])
                val = np.dot(interior_coeffs, interior_vals) + boundary_coeff * self.dirichlet_value

            else:   # neg_alpha is not None
                # Boundary in -y → interior rows go upward: offsets {0, ..., n_avail}
                n_avail = min(n, grid.y_points - 1 - j)
                row_offsets = np.arange(0, n_avail + 1, dtype=float)
                interior_coeffs, boundary_coeff = dirichlet_pseudo_boundary_backward(
                    neg_alpha / h, derivative_order, n_avail)
                interior_vals = np.array([
                    grid.values[(j + int(o)) * grid.x_points + i]
                    for o in row_offsets])
                val = boundary_coeff * self.dirichlet_value + np.dot(interior_coeffs, interior_vals)

            return val / (h ** derivative_order)

        else:
            raise ValueError(f"direction must be 'x' or 'y', got '{direction}'")

    # ------------------------------------------------------------------
    # K-P embedded boundary preprocessing (used by solve_leapfrog)
    # ------------------------------------------------------------------

    def preprocess_kp(self, grid, c_sq_values: np.ndarray, gamma: float = 0.25):
        """
        Precompute the Kreiss-Petersson modified Laplacian matrix A and
        boundary forcing function b_func for use with solve_leapfrog().

        Ghost points outside the domain are eliminated algebraically: their
        values are expressed as linear combinations of interior dof values
        plus the boundary condition, and substituted into the Laplacian rows
        of the near-boundary interior points.  The result is a precomputed
        sparse matrix A and time-dependent vector b(t) such that:

            u_tt ≈ A @ u_interior + b(t) + F(t)

        The γ-stabilization parameter prevents small-cell stiffness and
        guarantees that the CFL condition k/h < 1/√2 is preserved for γ ≥ 0.25.

        Parameters
        ----------
        grid        : Grid_2D
        c_sq_values : ndarray of shape (Ny, Nx), c²(x,y) at each grid point
        gamma       : stabilization parameter (must be ≥ 0.25)

        Returns
        -------
        A                : csr_matrix, shape (N_int, N_int)
        b_func           : callable(t: float) → ndarray of shape (N_int,)
        interior_mask    : bool ndarray of shape (Ny, Nx), True = active dof
        interior_to_full : int ndarray of shape (N_int,)
        full_to_interior : int ndarray of shape (Ny*Nx,), -1 for non-interior
        """
        cache_key = (grid.x_points, grid.y_points, gamma)
        if cache_key in self._kp_cache:
            return self._kp_cache[cache_key]

        Ny, Nx = grid.y_points, grid.x_points
        h_x = grid.x_grid.delta
        h_y = grid.y_grid.delta
        x_coords = grid.x_grid.x   # shape (Nx,)
        y_coords = grid.y_grid.x   # shape (Ny,)

        mask = self.get_mask(grid)  # (Ny, Nx), True = outside domain

        # ── Step 1: Classify all grid points ──────────────────────────
        # 0=interior, 1=near_boundary_interior, 2=ghost, 3=far_exterior
        AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        point_class = np.full((Ny, Nx), 3, dtype=np.int8)

        for j in range(Ny):
            for i in range(Nx):
                has_outside = False
                has_inside  = False
                for dj, di in AXIS_DIRS:
                    nj, ni = j + dj, i + di
                    if 0 <= nj < Ny and 0 <= ni < Nx:
                        if mask[nj, ni]:
                            has_outside = True
                        else:
                            has_inside = True
                if not mask[j, i]:
                    point_class[j, i] = 1 if has_outside else 0
                else:
                    point_class[j, i] = 2 if has_inside else 3

        interior_mask    = (point_class == 0) | (point_class == 1)
        interior_to_full = np.where(interior_mask.flatten())[0].astype(np.intp)
        full_to_interior = np.full(Ny * Nx, -1, dtype=np.intp)
        for k, flat in enumerate(interior_to_full):
            full_to_interior[flat] = k
        N_int = len(interior_to_full)

        # ── Step 2: Ghost point geometry ──────────────────────────────
        # For each ghost point: find x_Γ (closest boundary crossing),
        # compute inward normal, compute interpolation weights, and express
        # the ghost value as a linear combination of interior dofs + BC.
        ghost_data = {}  # flat_idx -> {'linear_coeffs': {full_idx: coeff},
                         #               'bc_coeff': float, 'xG': float, 'yG': float}

        for j in range(Ny):
            for i in range(Nx):
                if point_class[j, i] != 2:
                    continue

                flat_g = j * Nx + i
                x_g = x_coords[i]
                y_g = y_coords[j]

                # Find boundary crossings from each interior neighbor
                crossings = []
                for dj, di in AXIS_DIRS:
                    nj, ni = j + dj, i + di
                    if 0 <= nj < Ny and 0 <= ni < Nx and not mask[nj, ni]:
                        x_nb = x_coords[ni]
                        y_nb = y_coords[nj]
                        d_from_nb = self._find_boundary_distance(x_nb, y_nb, x_g, y_g)
                        seg_len = np.hypot(x_g - x_nb, y_g - y_nb)
                        t = d_from_nb / seg_len
                        x_cross = x_nb + t * (x_g - x_nb)
                        y_cross = y_nb + t * (y_g - y_nb)
                        d_to_ghost = seg_len - d_from_nb
                        crossings.append((d_to_ghost, x_cross, y_cross))

                if not crossings:
                    continue

                # Closest crossing → approximate boundary point x_Γ
                crossings.sort(key=lambda c: c[0])
                xi_Gamma, x_Gamma, y_Gamma = crossings[0]

                if xi_Gamma < 1e-15:
                    continue  # ghost point is on the boundary itself

                # Inward unit normal: from ghost toward x_Γ
                dx = x_Gamma - x_g
                dy = y_Gamma - y_g
                nx_n = dx / xi_Gamma
                ny_n = dy / xi_Gamma

                # Choose interpolation direction and compute grid-line data
                interp = self._kp_interp_data(
                    j, i, nx_n, ny_n, xi_Gamma,
                    h_x, h_y, x_coords, y_coords, Nx, Ny, full_to_interior
                )
                if interp is None:
                    continue  # couldn't find valid interpolation grid lines

                xi_I, flat_pts_I, weights_I, flat_pts_II, weights_II = interp

                # Normal-direction Lagrange weights with γ-stabilization
                g0, gI, gII = normal_lagrange_weights(xi_Gamma, xi_I)
                g0_stab  = g0  + gamma
                gI_stab  = gI  - 2.0 * gamma
                gII_stab = gII + gamma

                # Ghost value = (BC(t) - gI_stab*v_I - gII_stab*v_II) / g0_stab
                # Expand v_I and v_II to get linear_coeffs
                linear_coeffs = {}
                for flat_pt, w in zip(flat_pts_I, weights_I):
                    c = -(gI_stab / g0_stab) * w
                    if abs(c) > 1e-15:
                        linear_coeffs[int(flat_pt)] = linear_coeffs.get(int(flat_pt), 0.0) + c
                for flat_pt, w in zip(flat_pts_II, weights_II):
                    c = -(gII_stab / g0_stab) * w
                    if abs(c) > 1e-15:
                        linear_coeffs[int(flat_pt)] = linear_coeffs.get(int(flat_pt), 0.0) + c

                ghost_data[flat_g] = {
                    'linear_coeffs': linear_coeffs,
                    'bc_coeff':      1.0 / g0_stab,
                    'xG':            x_Gamma,
                    'yG':            y_Gamma,
                }

        # ── Step 3: Build sparse matrix A (N_int × N_int) ────────────
        # Scale for each neighbor direction: 1/h² per direction
        DIR_SCALE = {
            (-1, 0): 1.0 / h_y ** 2,   # -y neighbor
            ( 1, 0): 1.0 / h_y ** 2,   # +y neighbor
            ( 0,-1): 1.0 / h_x ** 2,   # -x neighbor
            ( 0, 1): 1.0 / h_x ** 2,   # +x neighbor
        }

        row_ids, col_ids, vals = [], [], []
        b_entries = []  # (interior_k, coeff, x_Gamma, y_Gamma)

        for k, flat in enumerate(interior_to_full):
            j, i = divmod(int(flat), Nx)
            c_sq_k = float(c_sq_values[j, i])
            center_coeff = 0.0

            for (dj, di), scale in DIR_SCALE.items():
                nj, ni = j + dj, i + di
                if not (0 <= nj < Ny and 0 <= ni < Nx):
                    continue  # off-grid: skip

                flat_nb  = nj * Nx + ni
                cls_nb   = int(point_class[nj, ni])

                if cls_nb in (0, 1):   # regular interior neighbor
                    int_idx = int(full_to_interior[flat_nb])
                    row_ids.append(k)
                    col_ids.append(int_idx)
                    vals.append(c_sq_k * scale)
                    center_coeff -= scale

                elif cls_nb == 2:      # ghost neighbor: substitute formula
                    center_coeff -= scale
                    if flat_nb not in ghost_data:
                        continue
                    gd = ghost_data[flat_nb]
                    for full_idx, lcoeff in gd['linear_coeffs'].items():
                        int_idx = int(full_to_interior[full_idx])
                        if int_idx >= 0:
                            row_ids.append(k)
                            col_ids.append(int_idx)
                            vals.append(c_sq_k * scale * lcoeff)
                    b_entries.append((k, c_sq_k * scale * gd['bc_coeff'],
                                      gd['xG'], gd['yG']))
                # cls_nb == 3 (far_exterior): never adjacent to interior by construction

            row_ids.append(k)
            col_ids.append(k)
            vals.append(c_sq_k * center_coeff)

        A = sp.coo_matrix((vals, (row_ids, col_ids)), shape=(N_int, N_int)).tocsr()

        # ── Step 4: Build b_func(t) ───────────────────────────────────
        if isinstance(self.dirichlet_value, (int, float)):
            dv = float(self.dirichlet_value)
            b_vec = np.zeros(N_int)
            for (k, coeff, xg, yg) in b_entries:
                b_vec[k] += coeff * dv
            b_func = lambda t, _b=b_vec: _b
        else:
            dv_fn = self.dirichlet_value
            b_entries_frozen = list(b_entries)
            def b_func(t):
                b = np.zeros(N_int)
                for (k, coeff, xg, yg) in b_entries_frozen:
                    b[k] += coeff * dv_fn(xg, yg, t)
                return b

        result = (A, b_func, interior_mask, interior_to_full, full_to_interior)
        self._kp_cache[cache_key] = result
        return result

    def _kp_interp_data(self, j, i, nx_n, ny_n, xi_Gamma,
                        h_x, h_y, x_coords, y_coords, Nx, Ny, full_to_interior):
        """
        Find two grid-line interpolation datasets for the K-P ghost point at (j,i).

        Returns (xi_I, flat_pts_I, weights_I, flat_pts_II, weights_II) or None.
        flat_pts_* are flat full-grid indices of the 3 interpolation points.
        """
        x_g = x_coords[i]
        y_g = y_coords[j]

        def get_line_x(j_line, x_query):
            """3-point quadratic interpolation along horizontal line j_line at x=x_query."""
            if not (0 <= j_line < Ny):
                return None
            i_c = int(np.argmin(np.abs(x_coords - x_query)))
            i_c = np.clip(i_c, 1, Nx - 2)
            cols = [i_c - 1, i_c, i_c + 1]
            flats = [j_line * Nx + ic for ic in cols]
            # All 3 must be interior dofs
            if any(full_to_interior[f] < 0 for f in flats):
                return None
            w = lagrange_weights(x_coords[cols], x_query)
            return flats, w

        def get_line_y(i_line, y_query):
            """3-point quadratic interpolation along vertical line i_line at y=y_query."""
            if not (0 <= i_line < Nx):
                return None
            j_c = int(np.argmin(np.abs(y_coords - y_query)))
            j_c = np.clip(j_c, 1, Ny - 2)
            rows = [j_c - 1, j_c, j_c + 1]
            flats = [jr * Nx + i_line for jr in rows]
            if any(full_to_interior[f] < 0 for f in flats):
                return None
            w = lagrange_weights(y_coords[rows], y_query)
            return flats, w

        # Case A: more vertical normal → horizontal grid lines
        if abs(ny_n) > abs(nx_n):
            j_sign = 1 if ny_n > 0 else -1
            j_I  = j + j_sign
            j_II = j + 2 * j_sign
            xi_I = h_y / abs(ny_n)
            x_I  = x_g + xi_I * nx_n
            x_II = x_g + 2.0 * xi_I * nx_n
            data_I  = get_line_x(j_I, x_I)
            data_II = get_line_x(j_II, x_II)
            if data_I is not None and data_II is not None:
                return xi_I, data_I[0], data_I[1], data_II[0], data_II[1]

        # Case B: more horizontal normal → vertical grid lines
        if abs(nx_n) < 1e-15:
            return None  # perfectly vertical normal but Case A failed
        i_sign = 1 if nx_n > 0 else -1
        i_I  = i + i_sign
        i_II = i + 2 * i_sign
        xi_I = h_x / abs(nx_n)
        y_I  = y_g + xi_I * ny_n
        y_II = y_g + 2.0 * xi_I * ny_n
        data_I  = get_line_y(i_I, y_I)
        data_II = get_line_y(i_II, y_II)
        if data_I is not None and data_II is not None:
            return xi_I, data_I[0], data_I[1], data_II[0], data_II[1]

        return None  # could not find valid interpolation data

class BoundaryConditions:
    def __init__(
        self,
        x_0_func: Optional[Callable] = None,
        x_L_func: Optional[Callable] = None,
        y_0_func: Optional[Callable] = None,
        y_L_func: Optional[Callable] = None,
        x_0_is_dirichlet: bool = True,
        x_L_is_dirichlet: bool = True,
        y_0_is_dirichlet: bool = True,
        y_L_is_dirichlet: bool = True,
    ):
        # Set default functions (zero boundary)
        default_func = lambda coords: np.zeros_like(coords)

        self.x_0_func = x_0_func if x_0_func is not None else default_func
        self.x_L_func = x_L_func if x_L_func is not None else default_func
        self.y_0_func = y_0_func if y_0_func is not None else default_func
        self.y_L_func = y_L_func if y_L_func is not None else default_func

        self.x_0_is_dirichlet = x_0_is_dirichlet
        self.x_L_is_dirichlet = x_L_is_dirichlet
        self.y_0_is_dirichlet = y_0_is_dirichlet
        self.y_L_is_dirichlet = y_L_is_dirichlet

        # Detect time dependency by checking function signatures
        self.is_time_dependent = self._check_time_dependency()

    def _check_time_dependency(self) -> bool:
        """
        Check if any boundary function accepts a time parameter.

        Returns True if any function has a signature accepting 2+ parameters
        (spatial_coords, time) or has 'time' in parameter names.
        """
        all_funcs = [self.x_0_func, self.x_L_func, self.y_0_func, self.y_L_func]

        for func in all_funcs:
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                # Check if function accepts 2+ parameters or has 'time' parameter
                if len(params) >= 2 or 'time' in params:
                    return True
            except (ValueError, TypeError):
                # Can't inspect (e.g., built-in function), assume time-independent
                continue

        return False

    @property
    def dirichlet_boundaries(self):
        boundaries = {}
        if self.x_0_is_dirichlet:
            boundaries['x_0'] = self.x_0_func
        if self.x_L_is_dirichlet:
            boundaries['x_L'] = self.x_L_func
        if self.y_0_is_dirichlet:
            boundaries['y_0'] = self.y_0_func
        if self.y_L_is_dirichlet:
            boundaries['y_L'] = self.y_L_func
        return boundaries

    @property
    def neumann_boundaries(self):
        boundaries = {}
        if not self.x_0_is_dirichlet:
            boundaries['x_0'] = self.x_0_func
        if not self.x_L_is_dirichlet:
            boundaries['x_L'] = self.x_L_func
        if not self.y_0_is_dirichlet:
            boundaries['y_0'] = self.y_0_func
        if not self.y_L_is_dirichlet:
            boundaries['y_L'] = self.y_L_func
        return boundaries

