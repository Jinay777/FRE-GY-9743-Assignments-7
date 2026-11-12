import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Shape-preserving C1 quadratic interpolator for instantaneous forward rates.

    The inputs are IFR point values f_i at node times t_i, so the curve must pass through
    them. On [t_0, t_N] the curve is built from quadratic pieces whose derivative is
    piecewise linear:

    1. Node slopes. At an interior node I take the harmonic mean of the two neighbouring
       secant slopes, 2 s_i s_{i+1} / (s_i + s_{i+1}), and 0 when the two secants have
       opposite signs or either is 0. When they share a sign this mean lies between them
       and never exceeds twice the smaller one. At the two end nodes I reflect the slope,
       d_0 = 2 s_1 - d_1 and d_N = 2 s_N - d_{N-1}, so each end interval is a single
       quadratic.
    2. Interval shape. On [a, b] with secant s and end slopes d0, d1, I insert a breakpoint
       xi (the analogue of Hagan and West's eta) and use one quadratic on each side. The
       slope D at xi follows from passing through f(b):
           D = 2 s - lam d0 - (1 - lam) d1,   lam = (xi - a) / (b - a).
       When d0 <= s <= d1 (or the reverse) I choose lam inside the range that keeps D
       between d0 and d1, so the slope moves one way across the interval: the piece is
       convex (or concave). Otherwise the data itself turns over here and I put xi at
       the midpoint.

    Guarantees:
    - Continuous and C1 inside [t_0, t_N], passing through every (t_i, f_i).
    - Monotone on every interval: every slope stays the same sign as s and between 0 and
      2 s, so the curve never overshoots its two node values (no spurious bumps or
      negative rates from positive data).
    - Convex over the whole node range whenever the data is convex (secant slopes
      non-decreasing), and concave whenever the data is concave.
    - FLAT extrapolation outside the nodes, and a one-node curve is a constant.

    When the data cannot satisfy every condition I keep continuity, interpolation and
    monotonicity, and give up convexity only where the data itself is not convex. In the
    edge case of exactly collinear points next to curved ones, a C1 convex curve does not
    exist, so I keep convexity and let the slope step up at that node instead.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        if self.extrap_method_ != ExtrapMethod.FLAT:
            raise ValueError("Monotone convex interpolation supports FLAT extrapolation only")
        self.t_ = np.asarray(axis1, dtype=float)
        self.f_ = np.asarray(values, dtype=float)
        if self.length_ > 1 and np.any(np.diff(self.t_) <= 0.0):
            raise ValueError("Monotone convex interpolation needs strictly increasing nodes")

        n = self.length_ - 1
        self.pieces_ = []
        self.node_integrals_ = np.zeros(self.length_)
        if n < 1:
            return

        h = np.diff(self.t_)
        secants = np.diff(self.f_) / h

        # Step 1: node slopes.
        slopes = np.zeros(self.length_)
        for i in range(1, n):
            left, right = secants[i - 1], secants[i]
            if left * right > 0.0:
                slopes[i] = 2.0 * left * right / (left + right)
        if n == 1:
            # Two nodes: I just join them with a straight line.
            slopes[0] = slopes[1] = secants[0]
        else:
            slopes[0] = 2.0 * secants[0] - slopes[1]
            slopes[n] = 2.0 * secants[n - 1] - slopes[n - 1]

        # Step 2: one or two quadratic pieces per interval.
        for i in range(n):
            a, b = self.t_[i], self.t_[i + 1]
            s, d0, d1 = secants[i], slopes[i], slopes[i + 1]
            lam = self._breakpoint_fraction(s, d0, d1)
            xi = a + lam * (b - a)
            knot_slope = 2.0 * s - lam * d0 - (1.0 - lam) * d1
            f_knot = self.f_[i] + (xi - a) * (d0 + knot_slope) / 2.0
            self.pieces_.append((a, xi, b, self.f_[i], f_knot, d0, knot_slope, d1))
            # Exact integral of the two quadratics over [a, b].
            self.node_integrals_[i + 1] = (self.node_integrals_[i]
                                           + self._piece_integral(i, b))

    @staticmethod
    def _breakpoint_fraction(s: float, d0: float, d1: float) -> float:
        # I measure each end slope against the secant: d0 = s - alpha, d1 = s + beta.
        alpha, beta = s - d0, d1 - s
        # I snap rounding noise to exactly zero, so collinear data is not misread as a turn.
        tol = 1e-12 * max(abs(s), abs(d0), abs(d1), 1e-300)
        alpha = 0.0 if abs(alpha) < tol else alpha
        beta = 0.0 if abs(beta) < tol else beta
        if alpha == 0.0 and beta == 0.0:
            return 0.5  # straight line, the breakpoint position is irrelevant
        if alpha * beta >= 0.0:
            # Locally convex (alpha, beta >= 0) or concave (both <= 0). Any lam in
            # [(beta - alpha) / (alpha + beta), 2 beta / (alpha + beta)] keeps the knot
            # slope between d0 and d1; I take the middle of that range inside [0, 1].
            lower = max(0.0, (beta - alpha) / (alpha + beta))
            upper = min(1.0, 2.0 * beta / (alpha + beta))
            return 0.5 * (lower + upper)
        # The data turns over inside this interval, so convexity cannot hold here.
        return 0.5

    def _piece_value(self, i: int, x: float) -> float:
        a, xi, b, fa, f_knot, d0, dk, d1 = self.pieces_[i]
        if x <= xi and xi > a:
            u = x - a
            return fa + d0 * u + (dk - d0) * u * u / (2.0 * (xi - a))
        u = x - xi
        return f_knot + dk * u + (d1 - dk) * u * u / (2.0 * (b - xi))

    def _piece_integral(self, i: int, x: float) -> float:
        # Integral of the curve from the start of interval i up to x (x inside the interval).
        a, xi, b, fa, f_knot, d0, dk, d1 = self.pieces_[i]
        total = 0.0
        if xi > a:
            u = min(x, xi) - a
            total += fa * u + d0 * u * u / 2.0 + (dk - d0) * u ** 3 / (6.0 * (xi - a))
        if x > xi:
            u = x - xi
            total += f_knot * u + dk * u * u / 2.0 + (d1 - dk) * u ** 3 / (6.0 * (b - xi))
        return total

    def _antiderivative(self, x: float) -> float:
        # F(x) = integral of the curve from t_0 to x, with the flat wings outside the nodes.
        t0, tn = self.t_[0], self.t_[-1]
        if x <= t0 or self.length_ == 1:
            return self.f_[0] * (x - t0)
        if x >= tn:
            return self.node_integrals_[-1] + self.f_[-1] * (x - tn)
        i = int(np.searchsorted(self.t_, x, side='right')) - 1
        return self.node_integrals_[i] + self._piece_integral(i, x)

    def interpolate(self, x: float) -> float:
        if self.length_ == 1 or x <= self.t_[0]:
            return float(self.f_[0])
        if x >= self.t_[-1]:
            return float(self.f_[-1])
        i = int(np.searchsorted(self.t_, x, side='right')) - 1
        return float(self._piece_value(i, x))

    def integrate(self, start_x: float, end_x: float) -> float:
        return float(self._antiderivative(end_x) - self._antiderivative(start_x))

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')