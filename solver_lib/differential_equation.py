
import sympy as sp
from typing import Dict, Tuple, Union

class DifferentialEquation:
    def __init__(self,
                 rhs,
                 u_symbol,
                 x_symbol,
                 y_symbol,
                 t_symbol,
                 lhs=None,
                 time_derivative_order: int = 1):
        
        self.rhs = rhs
        self.u_symbol = u_symbol
        self.x_symbol = x_symbol
        self.y_symbol = y_symbol
        self.t_symbol = t_symbol
        self.time_derivative_order = time_derivative_order

        # Set default LHS if not provided
        if lhs is None:
            # Default: du/dt for first-order, d2u/dt2 for second-order
            if time_derivative_order == 1:
                self.lhs = sp.diff(u_symbol, t_symbol)
            elif time_derivative_order == 2:
                self.lhs = sp.diff(u_symbol, t_symbol, t_symbol)
            else:
                raise ValueError(f"Unsupported time_derivative_order: {time_derivative_order}")
        else:
            self.lhs = lhs

        # Validate the equation
        self._validate_linearity()

        # Parse and extract terms (computed on demand)
        self._spatial_terms = None

    def _validate_linearity(self):
        # Expand the expression to separate additive terms
        expanded = sp.expand(self.rhs)
        terms = sp.Add.make_args(expanded)

        for term in terms:
            # Skip terms that don't involve u (source terms)
            if not term.has(self.u_symbol):
                continue

            # Count how many u-related factors appear in this term
            # We need to ensure at most one factor involves u or its derivatives

            # Get all multiplicative factors
            factors = sp.Mul.make_args(term)

            u_factor_count = 0
            for factor in factors:
                # Check if this factor involves u
                if factor.has(self.u_symbol):
                    u_factor_count += 1

                    # Check for nonlinear functions of u
                    # e.g., sin(u), exp(u), u**2, etc.
                    if not (factor == self.u_symbol or isinstance(factor, sp.Derivative)):
                        if isinstance(factor, sp.Pow) and factor.base == self.u_symbol:
                            if factor.exp != 1:
                                raise ValueError(
                                    f"Nonlinear term detected in RHS: {term}. "
                                    f"Found {self.u_symbol}**{factor.exp}, "
                                    f"but only linear terms are allowed."
                                )
                        else:
                            # Some other nonlinear function of u
                            raise ValueError(
                                f"Nonlinear term detected in RHS: {term}. "
                                f"Factor {factor} contains {self.u_symbol} in a nonlinear way."
                            )

            # Check for products of u-related terms (e.g., u * du/dx)
            if u_factor_count > 1:
                raise ValueError(
                    f"Nonlinear term detected in RHS: {term}. "
                    f"Found {u_factor_count} factors involving {self.u_symbol}. "
                    f"Products of u or its derivatives are not allowed in linear PDEs."
                )

    def extract_spatial_terms(self) -> Dict[Union[Tuple[int, int], str], sp.Expr]:
        # returns dict mapping (dx_order, dy_order) to coefficient expression

        # Return cached result if already computed
        if self._spatial_terms is not None:
            return self._spatial_terms

        term_dict = {}
        expanded = sp.expand(self.rhs)

        # Separate source terms (no u) from PDE terms
        source_terms = []
        pde_terms = sp.Add.make_args(expanded)

        # Build list of all derivative patterns to check
        # Check up to 4th order derivatives (sufficient for most PDEs)
        max_order = 4
        derivative_patterns = []

        # Generate all combinations: (dx_order, dy_order) where total order <= max_order
        for total_order in range(max_order, 0, -1):  # Check higher orders first
            for dx_order in range(total_order, -1, -1):
                dy_order = total_order - dx_order
                derivative_patterns.append((dx_order, dy_order))

        # Add the (0, 0) case for the u term itself
        derivative_patterns.append((0, 0))

        for term in pde_terms:
            if not term.has(self.u_symbol):
                source_terms.append(term)
                continue

            # Try to match this term against known derivative patterns
            matched = False

            for (dx_order, dy_order) in derivative_patterns:
                if dx_order == 0 and dy_order == 0:
                    # Check for u itself (no derivatives)
                    # We need to ensure this term doesn't contain any derivatives
                    has_derivatives = False
                    for other_dx, other_dy in derivative_patterns:
                        if other_dx == 0 and other_dy == 0:
                            continue
                        deriv = sp.diff(self.u_symbol,
                                      *([self.x_symbol]*other_dx + [self.y_symbol]*other_dy))
                        if term.has(deriv):
                            has_derivatives = True
                            break

                    if not has_derivatives and term.has(self.u_symbol):
                        # This is a pure u term
                        coeff = term.coeff(self.u_symbol)
                        term_dict[(0, 0)] = term_dict.get((0, 0), 0) + coeff
                        matched = True
                        break
                else:
                    # Check for specific derivative
                    deriv = sp.diff(self.u_symbol,
                                  *([self.x_symbol]*dx_order + [self.y_symbol]*dy_order))
                    if term.has(deriv):
                        coeff = term.coeff(deriv)
                        if coeff != 0:  # Only store non-zero coefficients
                            term_dict[(dx_order, dy_order)] = term_dict.get((dx_order, dy_order), 0) + coeff
                            matched = True
                            break

            if not matched:
                # This shouldn't happen if validation passed, but check anyway
                raise ValueError(f"Could not identify derivative pattern in term: {term}")

        # Combine source terms
        if source_terms:
            term_dict['source'] = sp.Add(*source_terms)

        # Cache the result
        self._spatial_terms = term_dict
        return term_dict

    def __repr__(self):
        return f"DifferentialEquation({self.lhs} = {self.rhs})"

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

    @property
    def is_parabolic(self) -> bool:
        return self.time_derivative_order == 1

    @property
    def is_hyperbolic(self) -> bool:
        return self.time_derivative_order == 2

    def get_coefficient(self, dx_order: int, dy_order: int) -> sp.Expr:
        terms = self.extract_spatial_terms()
        return terms.get((dx_order, dy_order), 0)

    def has_source_term(self) -> bool:
        terms = self.extract_spatial_terms()
        return 'source' in terms

    def get_source_term(self) -> sp.Expr:
        terms = self.extract_spatial_terms()
        return terms.get('source', 0)

def HeatEquation(alpha: float = 1.0) -> tuple["DifferentialEquation", tuple]:
    """Return (equation, (x, y, t, u)) for the 2-D heat equation.

    du/dt = alpha * (d²u/dx² + d²u/dy²)
    """
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    lhs = sp.diff(u(x, y), t)
    rhs = alpha * (sp.diff(u(x, y), x, x) + sp.diff(u(x, y), y, y))
    eq = DifferentialEquation(rhs=rhs, lhs=lhs, u_symbol=u(x, y),
                               x_symbol=x, y_symbol=y, t_symbol=t,
                               time_derivative_order=1)
    return eq, (x, y, t, u(x, y))


def WaveEquation(c, gamma: float = 0) -> tuple["DifferentialEquation", tuple]:
    """Return (equation, (x, y, t, u)) for the 2-D wave equation.

    d²u/dt² + gamma·du/dt = c²(d²u/dx² + d²u/dy²)
    """
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    lhs = sp.diff(u(x, y), t, t) + gamma * sp.diff(u(x, y), t)
    rhs = c ** 2 * (sp.diff(u(x, y), x, x) + sp.diff(u(x, y), y, y))
    eq = DifferentialEquation(rhs=rhs, lhs=lhs, u_symbol=u(x, y),
                               x_symbol=x, y_symbol=y, t_symbol=t,
                               time_derivative_order=2)
    return eq, (x, y, t, u(x, y))