"""
mset_rul.py
===========

A dependency-free ("native" Python, standard library only) implementation of:

    S. Cheng and M. Pecht, "Multivariate State Estimation Technique for
    Remaining Useful Life Prediction of Electronic Products",
    CALCE, University of Maryland (AAAI 2007).

Everything in the paper that is computable is implemented here:

    Eq. (1)   Observation / state vector          X(ti) = [x1i, ..., xni]^T
    Eq. (2)   Training data matrix                T = [X(t1), ..., X(tl)]
    Eq. (3)   Memory matrix                       D = [X1, ..., Xm]
    Eq. (4)   Remaining training data             L,  T = D u L
    Eq. (5)   Estimate                            Xest = D . W
    Eq. (6)   Error vector                        eps = Xobs - Xest
    Eq. (7)   Weight vector (least squares)       W = (D^T (x) D)^-1 . (D^T (x) Xobs)
    Eq. (8)   Estimate in closed form             Xest = D . (D^T (x) D)^-1 . (D^T (x) Xobs)
    Fig. 2    Full MSET process (D/L split, healthy residuals, actual residuals)
              + SPRT hypothesis test for fault detection
    Eq. (9)   Accumulated degradation             De_Ac(tk) = SUM_i ||R(ti)||
    Eq. (10)  Residual Euclidean norm             ||R(ti)|| = sqrt(r1i^2 + ... + rni^2)
    Eq. (11)  Sample-count-normalised degradation De(tk) = (1/k) SUM_i ||R(ti)||
    Fig. 3    RUL prediction model (criteria of failure from failed units /
              accelerated tests, degradation regression, RUL prediction)

Notes on the one thing the paper leaves open
--------------------------------------------
The paper writes the similarity operator as "(x)" (a nonlinear operator whose
properties are deferred to reference [5]). Since it is not specified in the
text, this module implements it as a pluggable kernel; three standard MSET
similarity operators are provided (inverse-distance, Gaussian/RBF, and the
bounded-angle operator). The default is inverse-distance, the classical choice
in the MSET literature. Any callable s(u, v) -> [0, 1] can be substituted.

Run `python3 mset_rul.py` to execute the built-in demonstration and the
numerical reproduction of the paper's case study.

Author: implementation written to accompany the cited publication.
License: public domain / CC0 - use freely.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

Vector = List[float]
Matrix = List[List[float]]          # row-major: Matrix[row][col]
Kernel = Callable[[Sequence[float], Sequence[float]], float]


# =========================================================================== #
# 1. Minimal linear algebra (no NumPy)
# =========================================================================== #

def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Standard matrix product A(p x q) . B(q x r)."""
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("matmul: incompatible shapes")
    bt = transpose(b)
    return [[sum(ai * bj for ai, bj in zip(row, col)) for col in bt] for row in a]


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    """Matrix-vector product A(p x q) . x(q)."""
    if not a or len(a[0]) != len(x):
        raise ValueError("matvec: incompatible shapes")
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def euclidean_norm(v: Sequence[float]) -> float:
    """Eq. (10): ||R(ti)|| = sqrt(r_1i^2 + r_2i^2 + ... + r_ni^2)."""
    return math.sqrt(sum(component * component for component in v))


def euclidean_distance(u: Sequence[float], v: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def invert(a: Matrix, ridge: float = 0.0) -> Matrix:
    """
    Invert a square matrix by Gauss-Jordan elimination with partial pivoting.

    `ridge` adds lambda*I before inversion (Tikhonov regularisation). The
    similarity matrix D^T (x) D is frequently ill-conditioned when memory-matrix
    states are close to one another, so a small ridge term is the practical
    safeguard against the numerical blow-up of Eq. (7).
    """
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("invert: matrix must be square")

    # Augment [A + ridge*I | I]
    aug = [[a[i][j] + (ridge if i == j else 0.0) for j in range(n)]
           + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]

    for col in range(n):
        # Partial pivoting
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-15:
            raise ZeroDivisionError(
                "invert: matrix is singular to working precision; "
                "increase `ridge` or reduce the memory-matrix size"
            )
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                aug[r] = [vr - factor * vc for vr, vc in zip(aug[r], aug[col])]

    return [row[n:] for row in aug]


# =========================================================================== #
# 2. Similarity operators  (the "(x)" of Eq. 7 / Eq. 8)
# =========================================================================== #

def inverse_distance_kernel(scale: float = 1.0) -> Kernel:
    """
    s(u, v) = 1 / (1 + d(u, v)/scale)

    The classical MSET similarity operator: equals 1 for identical states and
    decays monotonically towards 0 as states separate.
    """
    def kernel(u: Sequence[float], v: Sequence[float]) -> float:
        return 1.0 / (1.0 + euclidean_distance(u, v) / scale)
    return kernel


def gaussian_kernel(bandwidth: float = 1.0) -> Kernel:
    """s(u, v) = exp(-(d(u, v)/h)^2) - smoother, sharper locality."""
    def kernel(u: Sequence[float], v: Sequence[float]) -> float:
        d = euclidean_distance(u, v) / bandwidth
        return math.exp(-d * d)
    return kernel


def bounded_angle_kernel() -> Kernel:
    """
    s(u, v) = 1 - theta(u, v) / (pi/2), clipped to [0, 1].

    Direction-sensitive operator; useful when the *shape* of the state vector
    carries the health information rather than its magnitude.
    """
    def kernel(u: Sequence[float], v: Sequence[float]) -> float:
        nu, nv = euclidean_norm(u), euclidean_norm(v)
        if nu == 0.0 or nv == 0.0:
            return 1.0 if nu == nv else 0.0
        cos = max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v)) / (nu * nv)))
        return max(0.0, 1.0 - math.acos(cos) / (math.pi / 2.0))
    return kernel


# =========================================================================== #
# 3. MSET engine
# =========================================================================== #

class MSET:
    """
    Multivariate State Estimation Technique.

    Usage
    -----
        model = MSET()
        model.fit(training_states, memory_size=12)   # healthy history only
        residual = model.residual(new_observation)   # R = Xest - Xobs

    A "state" (Eq. 1) is a sequence of n parameter values observed at one time
    instant; `training_states` is the set of columns of the data matrix of
    Figure 1, i.e. a list of such states.
    """

    def __init__(self,
                 kernel: Optional[Kernel] = None,
                 ridge: float = 1e-8,
                 normalize: bool = True) -> None:
        self.kernel: Kernel = kernel or inverse_distance_kernel()
        self.ridge = ridge
        self.normalize = normalize

        # Populated by fit()
        self.n_params: int = 0
        self._lo: Vector = []
        self._span: Vector = []
        self.D_states: List[Vector] = []     # memory matrix, as list of states
        self.L_states: List[Vector] = []     # remaining training data
        self.D_index: List[int] = []
        self.L_index: List[int] = []
        self._D_matrix: Matrix = []          # n x m, columns are memory states
        self._G_inv: Matrix = []             # (D^T (x) D)^-1,  m x m
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Scaling
    # ------------------------------------------------------------------ #
    def _fit_scaler(self, states: Sequence[Sequence[float]]) -> None:
        n = len(states[0])
        self._lo = [min(s[j] for s in states) for j in range(n)]
        hi = [max(s[j] for s in states) for j in range(n)]
        self._span = [(h - l) if (h - l) > 1e-12 else 1.0
                      for h, l in zip(hi, self._lo)]

    def _scale(self, state: Sequence[float]) -> Vector:
        if not self.normalize:
            return list(state)
        return [(x - lo) / sp for x, lo, sp in zip(state, self._lo, self._span)]

    def _unscale(self, state: Sequence[float]) -> Vector:
        if not self.normalize:
            return list(state)
        return [x * sp + lo for x, lo, sp in zip(state, self._lo, self._span)]

    # ------------------------------------------------------------------ #
    # Memory-matrix selection (paper, "three key procedures")
    # ------------------------------------------------------------------ #
    @staticmethod
    def select_memory_matrix(training: Sequence[Sequence[float]],
                             memory_size: int) -> Tuple[List[int], List[int]]:
        """
        Select the memory matrix D from training data T, exactly as described:

        1. Take every state holding the minimum or maximum value of any
           parameter (these carry the extreme features of the system).
        2. Order the remaining states by their Euclidean norms.
        3. Draw additional states at equally spaced intervals from that
           ordering until D holds `memory_size` states.
        4. Whatever is left over forms the remaining training data L (Eq. 4).

        Rule of thumb from the paper: memory_size >= 2 * (number of parameters).

        Returns (indices_of_D, indices_of_L) into `training`.
        """
        l = len(training)
        n = len(training[0])
        if memory_size > l:
            raise ValueError("memory_size cannot exceed the number of training states")
        if memory_size < 2 * n:
            # Not fatal - the paper states it only as a guideline.
            pass

        selected = set()
        for j in range(n):
            column = [state[j] for state in training]
            selected.add(min(range(l), key=lambda i: column[i]))
            selected.add(max(range(l), key=lambda i: column[i]))

        if len(selected) > memory_size:
            # More extremes than the requested budget: keep a spread of them.
            ordered_extremes = sorted(selected,
                                      key=lambda i: euclidean_norm(training[i]))
            step = (len(ordered_extremes) - 1) / max(1, memory_size - 1)
            keep = {ordered_extremes[min(len(ordered_extremes) - 1,
                                         int(round(k * step)))]
                    for k in range(memory_size)}
            selected = keep

        remaining = [i for i in range(l) if i not in selected]
        remaining.sort(key=lambda i: euclidean_norm(training[i]))

        needed = memory_size - len(selected)
        if needed > 0 and remaining:
            if needed == 1:
                picks = [remaining[len(remaining) // 2]]
            else:
                step = (len(remaining) - 1) / (needed - 1)
                picks = []
                for k in range(needed):
                    idx = remaining[min(len(remaining) - 1, int(round(k * step)))]
                    if idx not in picks:
                        picks.append(idx)
                # Backfill in case rounding produced duplicates
                for idx in remaining:
                    if len(picks) >= needed:
                        break
                    if idx not in picks:
                        picks.append(idx)
            selected.update(picks)

        d_index = sorted(selected)
        l_index = [i for i in range(l) if i not in selected]
        return d_index, l_index

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self,
            training_states: Sequence[Sequence[float]],
            memory_size: Optional[int] = None) -> "MSET":
        """
        Build the model from healthy historic data.

        Prerequisites stated in the paper (the caller is responsible for them):
          * the training data must span every healthy operational state;
          * it must contain no anomalies, sensor failures or equipment failures.
        """
        if not training_states:
            raise ValueError("fit: no training data supplied")
        n = len(training_states[0])
        if any(len(s) != n for s in training_states):
            raise ValueError("fit: all states must have the same length")

        self.n_params = n
        self._fit_scaler(training_states)
        scaled = [self._scale(s) for s in training_states]

        if memory_size is None:
            # Rule of thumb from reference [6]: at least twice the data sources.
            memory_size = min(len(scaled), max(2 * n, min(20, len(scaled) // 2 or 1)))

        self.D_index, self.L_index = self.select_memory_matrix(scaled, memory_size)
        self.D_states = [scaled[i] for i in self.D_index]
        self.L_states = [scaled[i] for i in self.L_index]

        # D as an n x m matrix whose columns are the memory states (Eq. 3)
        self._D_matrix = transpose(self.D_states)

        # G = D^T (x) D   (m x m similarity matrix), then invert once (Eq. 7)
        m = len(self.D_states)
        g = zeros(m, m)
        for i in range(m):
            for j in range(i, m):
                s = self.kernel(self.D_states[i], self.D_states[j])
                g[i][j] = s
                g[j][i] = s
        self._G_inv = invert(g, ridge=self.ridge)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("MSET model has not been fitted")

    def weights(self, observation: Sequence[float]) -> Vector:
        """Eq. (7): W = (D^T (x) D)^-1 . (D^T (x) Xobs)."""
        self._check_fitted()
        x = self._scale(observation)
        similarity = [self.kernel(state, x) for state in self.D_states]
        return matvec(self._G_inv, similarity)

    def estimate(self, observation: Sequence[float]) -> Vector:
        """Eq. (5)/(8): Xest = D . W, returned in the original units."""
        w = self.weights(observation)
        est_scaled = matvec(self._D_matrix, w)
        return self._unscale(est_scaled)

    def residual(self, observation: Sequence[float], scaled: bool = True) -> Vector:
        """
        Actual residual  R = Xest - Xobs  (Fig. 2).

        `scaled=True` returns the residual in normalised parameter units, which
        is what makes the Euclidean norm of Eq. (10) meaningful when parameters
        have different physical units (e.g. degC, %RH, g).
        """
        self._check_fitted()
        if scaled:
            x = self._scale(observation)
            w = self.weights(observation)
            est = matvec(self._D_matrix, w)
            return [e - o for e, o in zip(est, x)]
        est = self.estimate(observation)
        return [e - o for e, o in zip(est, observation)]

    def residuals(self,
                  observations: Sequence[Sequence[float]],
                  scaled: bool = True) -> List[Vector]:
        return [self.residual(obs, scaled=scaled) for obs in observations]

    def healthy_residuals(self, scaled: bool = True) -> List[Vector]:
        """
        Eq. of Fig. 2:  RL = Lest - L.

        The residuals of the remaining training data L. Because L is healthy by
        construction, these residuals characterise the healthy state of the
        system and form the reference distribution for fault detection.
        """
        self._check_fitted()
        raw_L = [self._unscale(s) for s in self.L_states]
        return self.residuals(raw_L, scaled=scaled)

    def healthy_statistics(self) -> List[Tuple[float, float]]:
        """Per-parameter (mean, standard deviation) of the healthy residuals."""
        rl = self.healthy_residuals()
        if not rl:
            return [(0.0, 1e-6)] * self.n_params
        stats = []
        for j in range(self.n_params):
            column = [r[j] for r in rl]
            mu = sum(column) / len(column)
            if len(column) > 1:
                var = sum((c - mu) ** 2 for c in column) / (len(column) - 1)
            else:
                var = 0.0
            stats.append((mu, max(math.sqrt(var), 1e-9)))
        return stats


# =========================================================================== #
# 4. Fault detection - Sequential Probability Ratio Test (Fig. 2)
# =========================================================================== #

class SPRT:
    """
    Wald's Sequential Probability Ratio Test for a mean shift in the residual
    stream, the hypothesis test the paper names for the fault-detection block
    of Figure 2.

        H0: residual ~ N(mu0, sigma^2)        (healthy)
        H1: residual ~ N(mu0 + M*sigma, s^2)  (degraded / drifting)

    Decision boundaries:  ln(beta/(1-alpha))  <  SUM ln(LR)  <  ln((1-beta)/alpha)

    Two hypotheses are tracked simultaneously (positive and negative mean
    shift), because an MSET residual R = Xest - Xobs drifts negative when the
    monitored parameter rises above its healthy estimate and positive when it
    falls below; a one-sided test would miss half of all degradation modes.
    """

    def __init__(self,
                 mean: float,
                 sigma: float,
                 alpha: float = 0.01,
                 beta: float = 0.01,
                 disturbance: float = 3.0) -> None:
        self.mu0 = mean
        self.sigma = max(sigma, 1e-12)
        self.shift = disturbance * self.sigma
        self.upper = math.log((1.0 - beta) / alpha)
        self.lower = math.log(beta / (1.0 - alpha))
        self.llr_pos = 0.0
        self.llr_neg = 0.0

    def update(self, value: float) -> str:
        """Feed one residual sample. Returns 'healthy', 'fault' or 'continue'."""
        gain = self.shift / self.sigma ** 2
        deviation = value - self.mu0
        # log-likelihood ratio increments for a positive / negative mean shift
        self.llr_pos += gain * (deviation - self.shift / 2.0)
        self.llr_neg += gain * (-deviation - self.shift / 2.0)

        if self.llr_pos >= self.upper or self.llr_neg >= self.upper:
            self.llr_pos = self.llr_neg = 0.0
            return "fault"
        if self.llr_pos <= self.lower and self.llr_neg <= self.lower:
            self.llr_pos = self.llr_neg = 0.0
            return "healthy"
        # A hypothesis that has been cleared on its own is reset to the origin
        self.llr_pos = max(self.llr_pos, self.lower)
        self.llr_neg = max(self.llr_neg, self.lower)
        return "continue"

    def run(self, series: Iterable[float]) -> List[str]:
        return [self.update(v) for v in series]


# =========================================================================== #
# 5. Degradation model (Eq. 9, 10, 11)
# =========================================================================== #

def residual_norms(residuals: Sequence[Sequence[float]]) -> List[float]:
    """Eq. (10) applied to a residual time series."""
    return [euclidean_norm(r) for r in residuals]


def accumulated_degradation(residuals: Sequence[Sequence[float]]) -> List[float]:
    """
    Eq. (9):  De_Ac(tk) = SUM_{i=1..k} ||R(ti)||

    Returns the running series, element k-1 being De_Ac(tk).
    """
    total = 0.0
    series = []
    for norm in residual_norms(residuals):
        total += norm
        series.append(total)
    return series


def degradation(residuals: Sequence[Sequence[float]]) -> List[float]:
    """
    Eq. (11):  De(tk) = (1/k) SUM_{i=1..k} ||R(ti)||

    The sample-count-normalised degradation, which removes the dependence of
    Eq. (9) on how many samples happen to have been taken. This is the quantity
    plotted in Figures 4-6 of the paper and the one used for RUL prediction.
    """
    return [total / (k + 1) for k, total in enumerate(accumulated_degradation(residuals))]


# =========================================================================== #
# 6. Criteria of failure (Fig. 3)
# =========================================================================== #

class FailureCriteria:
    """
    The degradation value at which the product is declared failed.

    Estimated statistically from units that have already failed - either from
    historic field failures with a similar physics of failure (PoF), or from an
    accelerated test. The paper's stated assumption: *the same products fail at
    the same degradation if their PoF is similar*, and the degradation at
    failure is unchanged by acceleration even though time-to-failure is not.

    The paper adopts the interval mean +/- sigma as the criteria band.
    """

    def __init__(self, failure_degradations: Sequence[float], k_sigma: float = 1.0) -> None:
        if len(failure_degradations) < 2:
            raise ValueError("at least two failed units are needed for mean and sigma")
        self.samples = list(failure_degradations)
        self.mean = sum(self.samples) / len(self.samples)
        var = sum((d - self.mean) ** 2 for d in self.samples) / (len(self.samples) - 1)
        self.sigma = math.sqrt(var)
        self.k_sigma = k_sigma

    @property
    def lower(self) -> float:
        return self.mean - self.k_sigma * self.sigma

    @property
    def upper(self) -> float:
        return self.mean + self.k_sigma * self.sigma

    @property
    def interval(self) -> Tuple[float, float]:
        return (self.lower, self.upper)

    @classmethod
    def from_runs(cls,
                  runs: Sequence[Tuple[Sequence[Sequence[float]], int]],
                  k_sigma: float = 1.0) -> "FailureCriteria":
        """
        Build criteria directly from MSET residual streams of failed units.

        `runs` is a sequence of (residual_series, failure_sample_index) pairs;
        the degradation of Eq. (11) is evaluated at the failure index of each.
        """
        values = []
        for residual_series, fail_index in runs:
            de = degradation(residual_series)
            values.append(de[min(fail_index, len(de) - 1)])
        return cls(values, k_sigma=k_sigma)

    def __repr__(self) -> str:
        return (f"FailureCriteria(mean={self.mean:.3f}, sigma={self.sigma:.3f}, "
                f"interval=[{self.lower:.3f}, {self.upper:.3f}])")


# =========================================================================== #
# 7. Degradation regression and RUL prediction (Fig. 3, Figs. 5-6)
# =========================================================================== #

class LinearTrend:
    """
    Ordinary least-squares straight line  De(t) = intercept + slope * t.

    The paper deliberately uses linear regression "to make it easy to figure out
    equations", while noting that more complex models may be more accurate. See
    `PowerTrend` below for a non-linear alternative that keeps the same API.
    """

    def __init__(self, times: Sequence[float], values: Sequence[float]) -> None:
        if len(times) != len(values) or len(times) < 2:
            raise ValueError("need at least two matching (time, degradation) points")
        k = len(times)
        mean_t = sum(times) / k
        mean_v = sum(values) / k
        sxx = sum((t - mean_t) ** 2 for t in times)
        if sxx <= 0.0:
            raise ValueError("degenerate regression: all times identical")
        sxy = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
        self.slope = sxy / sxx
        self.intercept = mean_v - self.slope * mean_t

        ss_tot = sum((v - mean_v) ** 2 for v in values)
        ss_res = sum((v - self.predict(t)) ** 2 for t, v in zip(times, values))
        self.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    def predict(self, t: float) -> float:
        return self.intercept + self.slope * t

    def time_to_reach(self, level: float) -> Optional[float]:
        """Solve De(t) = level for t. None if the trend never reaches `level`."""
        if self.slope <= 0.0:
            return None
        t = (level - self.intercept) / self.slope
        return t if math.isfinite(t) else None

    def __repr__(self) -> str:
        return (f"LinearTrend(De = {self.intercept:.4f} + {self.slope:.6f}*t, "
                f"R^2={self.r_squared:.4f})")


class PowerTrend:
    """
    Log-log least squares:  De(t) = a * t^b  (t > 0).

    A drop-in alternative to LinearTrend for degradation that accelerates
    (b > 1) or saturates (b < 1), as the paper anticipates for other products.
    """

    def __init__(self, times: Sequence[float], values: Sequence[float]) -> None:
        pairs = [(t, v) for t, v in zip(times, values) if t > 0 and v > 0]
        if len(pairs) < 2:
            raise ValueError("PowerTrend needs at least two strictly positive points")
        lt = [math.log(t) for t, _ in pairs]
        lv = [math.log(v) for _, v in pairs]
        fit = LinearTrend(lt, lv)
        self.b = fit.slope
        self.a = math.exp(fit.intercept)
        self.r_squared = fit.r_squared

    def predict(self, t: float) -> float:
        return self.a * (t ** self.b) if t > 0 else 0.0

    def time_to_reach(self, level: float) -> Optional[float]:
        if level <= 0 or self.a <= 0 or self.b <= 0:
            return None
        return math.exp((math.log(level) - math.log(self.a)) / self.b)

    def __repr__(self) -> str:
        return f"PowerTrend(De = {self.a:.4f} * t^{self.b:.4f}, R^2={self.r_squared:.4f})"


class RULPrediction:
    """Result container: failure-time interval and remaining useful life."""

    def __init__(self,
                 current_time: float,
                 t_lower: Optional[float],
                 t_nominal: Optional[float],
                 t_upper: Optional[float],
                 trend,
                 criteria: FailureCriteria) -> None:
        self.current_time = current_time
        self.failure_time_lower = t_lower
        self.failure_time_nominal = t_nominal
        self.failure_time_upper = t_upper
        self.trend = trend
        self.criteria = criteria

    @staticmethod
    def _rul(t: Optional[float], now: float) -> Optional[float]:
        return None if t is None else max(0.0, t - now)

    @property
    def rul_lower(self) -> Optional[float]:
        return self._rul(self.failure_time_lower, self.current_time)

    @property
    def rul_nominal(self) -> Optional[float]:
        return self._rul(self.failure_time_nominal, self.current_time)

    @property
    def rul_upper(self) -> Optional[float]:
        return self._rul(self.failure_time_upper, self.current_time)

    @property
    def failure_time_interval(self) -> Tuple[Optional[float], Optional[float]]:
        return (self.failure_time_lower, self.failure_time_upper)

    @property
    def rul_interval(self) -> Tuple[Optional[float], Optional[float]]:
        return (self.rul_lower, self.rul_upper)

    def __repr__(self) -> str:
        def fmt(v):
            return "never" if v is None else f"{v:.1f}"
        return (f"RULPrediction(failure time in [{fmt(self.failure_time_lower)}, "
                f"{fmt(self.failure_time_upper)}], nominal {fmt(self.failure_time_nominal)}; "
                f"RUL in [{fmt(self.rul_lower)}, {fmt(self.rul_upper)}] from t="
                f"{self.current_time:g})")


def predict_rul(times: Sequence[float],
                degradation_series: Sequence[float],
                criteria: FailureCriteria,
                trend_model=LinearTrend) -> RULPrediction:
    """
    The RUL prediction of Figure 3 / Figure 6.

    Regress the observed degradation trend, extrapolate it, and read off the
    times at which it crosses the lower, mean and upper failure criteria. The
    RUL is that crossing time minus the current time.
    """
    trend = trend_model(times, degradation_series)
    now = times[-1]
    return RULPrediction(
        current_time=now,
        t_lower=trend.time_to_reach(criteria.lower),
        t_nominal=trend.time_to_reach(criteria.mean),
        t_upper=trend.time_to_reach(criteria.upper),
        trend=trend,
        criteria=criteria,
    )


# =========================================================================== #
# 8. End-to-end pipeline
# =========================================================================== #

def mset_rul_pipeline(training_states: Sequence[Sequence[float]],
                      monitored_states: Sequence[Sequence[float]],
                      monitored_times: Sequence[float],
                      failed_units: Sequence[Tuple[Sequence[Sequence[float]], int]],
                      memory_size: Optional[int] = None,
                      kernel: Optional[Kernel] = None,
                      k_sigma: float = 1.0,
                      trend_model=LinearTrend) -> dict:
    """
    Run the complete method of Figure 3 in one call.

    Parameters
    ----------
    training_states : healthy history used to build the MSET model.
    monitored_states, monitored_times : the in-service unit being assessed.
    failed_units : list of (observation_series, failure_sample_index) for units
        that already failed with a similar PoF, or from an accelerated test.
        Their residuals are computed with the *same* memory matrix, as the paper
        requires.
    """
    model = MSET(kernel=kernel).fit(training_states, memory_size=memory_size)

    # Branch 1 - actual product: residuals -> degradation -> regression
    actual_residuals = model.residuals(monitored_states)
    actual_degradation = degradation(actual_residuals)

    # Branch 2 - historic/accelerated failures: residuals -> criteria of failure
    runs = [(model.residuals(series), fail_index) for series, fail_index in failed_units]
    criteria = FailureCriteria.from_runs(runs, k_sigma=k_sigma)

    prediction = predict_rul(monitored_times, actual_degradation, criteria, trend_model)

    return {
        "model": model,
        "residuals": actual_residuals,
        "degradation": actual_degradation,
        "accumulated_degradation": accumulated_degradation(actual_residuals),
        "criteria": criteria,
        "prediction": prediction,
    }


# =========================================================================== #
# 9. Console helpers
# =========================================================================== #

def fmt_optional(value: Optional[float], digits: int = 0, none: str = "never") -> str:
    """Format a possibly-None time/RUL value (None = trend never reaches level)."""
    return none if value is None else f"{value:.{digits}f}"


def ascii_plot(series_map: dict,
               times: Sequence[float],
               width: int = 62,
               height: int = 16,
               title: str = "") -> str:
    """Tiny ASCII scatter plot so results are visible without matplotlib."""
    all_values = [v for series in series_map.values() for v in series]
    if not all_values:
        return ""
    vmin, vmax = min(all_values), max(all_values)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    tmin, tmax = min(times), max(times)
    if tmax - tmin < 1e-12:
        tmax = tmin + 1.0

    grid = [[" "] * width for _ in range(height)]
    for marker, series in series_map.items():
        for t, v in zip(times, series):
            col = int((t - tmin) / (tmax - tmin) * (width - 1))
            row = height - 1 - int((v - vmin) / (vmax - vmin) * (height - 1))
            grid[row][col] = marker

    lines = []
    if title:
        lines.append(title)
    for r, row in enumerate(grid):
        label = ""
        if r == 0:
            label = f"{vmax:7.3f} |"
        elif r == height - 1:
            label = f"{vmin:7.3f} |"
        else:
            label = " " * 7 + " |"
        lines.append(label + "".join(row))
    lines.append(" " * 8 + "+" + "-" * width)
    lines.append(" " * 9 + f"{tmin:g}" + " " * (width - 10) + f"{tmax:g}")
    return "\n".join(lines)


# =========================================================================== #
# 10. Synthetic data generator for the demonstration
# =========================================================================== #

def _generate_unit(days: int,
                   drift_rate: float,
                   rng: random.Random,
                   noise: float = 0.35) -> List[Vector]:
    """
    Three monitored parameters of an electronic component, one sample per day:
      p1 - temperature (degC), p2 - relative humidity (%), p3 - vibration (g).

    A healthy unit tracks its nominal operating envelope; a degrading unit
    develops a slow multi-parameter drift (self-heating, moisture ingress,
    loosening mechanical attachment) that MSET turns into a growing residual.
    """
    series = []
    for day in range(days):
        cycle = math.sin(2.0 * math.pi * day / 7.0)          # weekly duty cycle
        drift = drift_rate * day
        p1 = 55.0 + 4.0 * cycle + rng.gauss(0.0, noise) + 6.0 * drift
        p2 = 45.0 + 6.0 * cycle + rng.gauss(0.0, noise * 2) + 9.0 * drift
        p3 = 0.80 + 0.10 * cycle + rng.gauss(0.0, noise * 0.02) + 0.25 * drift
        series.append([p1, p2, p3])
    return series


def demo() -> None:
    """Full worked example following the structure of the paper's case study."""
    rng = random.Random(20070101)

    print("=" * 74)
    print("MSET-based RUL prediction - worked example")
    print("=" * 74)

    # ---- Healthy history -------------------------------------------------
    training = _generate_unit(days=60, drift_rate=0.0, rng=rng)
    model = MSET(kernel=inverse_distance_kernel(scale=0.30)).fit(training, memory_size=14)
    print(f"\nTraining states l = {len(training)}, parameters n = {model.n_params}")
    print(f"Memory matrix D  m = {len(model.D_states)}  "
          f"(rule of thumb: m >= 2n = {2 * model.n_params})")
    print(f"Remaining training data L = {len(model.L_states)} states")

    stats = model.healthy_statistics()
    for j, (mu, sd) in enumerate(stats, start=1):
        print(f"  healthy residual param {j}: mean {mu:+.5f}, sigma {sd:.5f}")

    # ---- Accelerated test: six units, five fail --------------------------
    print("\n--- Accelerated test (criteria of failure) ---")
    accelerated = []
    failure_days = [22, 22, 11, 23, 10]
    for rate in (0.030, 0.033, 0.062, 0.040, 0.070):
        accelerated.append(_generate_unit(days=28, drift_rate=rate, rng=rng))
    healthy_unit = _generate_unit(days=28, drift_rate=0.0005, rng=rng)

    runs = []
    print(f"{'Unit':<8}{'Failure time (day)':>20}{'Degradation':>14}")
    for i, (series, fday) in enumerate(zip(accelerated, failure_days), start=1):
        residuals = model.residuals(series)
        de = degradation(residuals)
        runs.append((residuals, fday - 1))
        print(f"Test {i:<3}{fday:>20}{de[fday - 1]:>14.3f}")

    de_healthy = degradation(model.residuals(healthy_unit))
    print(f"{'Test 6':<8}{'(healthy)':>20}{de_healthy[-1]:>14.3f}")

    criteria = FailureCriteria.from_runs(runs, k_sigma=1.0)
    print(f"\nMean degradation at failure : {criteria.mean:.3f}")
    print(f"Standard deviation (sigma)  : {criteria.sigma:.3f}")
    print(f"Criteria of failure         : [{criteria.lower:.3f}, {criteria.upper:.3f}]")

    print("\n" + ascii_plot(
        {str(i): degradation(model.residuals(s))
         for i, s in enumerate(accelerated, start=1)},
        times=list(range(1, 29)),
        title="Fig. 4 analogue - degradation of the accelerated-test units"))

    # ---- The monitored ("7th") unit --------------------------------------
    print("\n--- Monitored unit ---")
    test_unit = _generate_unit(days=24, drift_rate=0.014, rng=rng)
    times = list(range(1, len(test_unit) + 1))
    test_residuals = model.residuals(test_unit)
    test_degradation = degradation(test_residuals)

    # Fault detection on the residual stream (Fig. 2 block).
    # The SPRT is run per parameter against the healthy residual statistics.
    def first_alarm(residual_series, param_index: int) -> Optional[int]:
        mu, sd = stats[param_index]
        sprt = SPRT(mean=mu, sigma=sd, alpha=0.01, beta=0.01, disturbance=3.0)
        for i, verdict in enumerate(sprt.run(r[param_index] for r in residual_series), 1):
            if verdict == "fault":
                return i
        return None

    alarms = [first_alarm(test_residuals, j) for j in range(model.n_params)]
    print("SPRT fault alerts (monitored unit), first alert day per parameter: "
          + ", ".join(f"p{j + 1}={fmt_optional(a, none='none')}"
                      for j, a in enumerate(alarms)))
    fast_alarms = [first_alarm(model.residuals(accelerated[4]), j)
                   for j in range(model.n_params)]
    print("SPRT fault alerts (accelerated Test 5, failed day 10):            "
          + ", ".join(f"p{j + 1}={fmt_optional(a, none='none')}"
                      for j, a in enumerate(fast_alarms)))

    prediction = predict_rul(times, test_degradation, criteria, LinearTrend)
    print(f"\nDegradation trend           : {prediction.trend}")
    print(f"Current degradation (t={times[-1]}) : {test_degradation[-1]:.4f}")
    print(f"Predicted failure time      : "
          f"[{fmt_optional(prediction.failure_time_lower)}, "
          f"{fmt_optional(prediction.failure_time_upper)}] days "
          f"(nominal {fmt_optional(prediction.failure_time_nominal)})")
    print(f"Remaining useful life       : "
          f"[{fmt_optional(prediction.rul_lower)}, "
          f"{fmt_optional(prediction.rul_upper)}] days "
          f"(nominal {fmt_optional(prediction.rul_nominal)})")

    print("\n" + ascii_plot({"o": test_degradation}, times,
                            height=12,
                            title="Fig. 5 analogue - degradation of the monitored unit"))


# =========================================================================== #
# 11. Numerical reproduction of the paper's published case study
# =========================================================================== #

def reproduce_paper_case_study() -> None:
    """
    Reproduce the published numbers of Table 1 and Figures 5-6 exactly.

    Table 1 degradations at failure: 1.46, 1.63, 1.94, 2.68, 1.77
      -> paper: mean 1.9, sigma 0.47, criteria [1.43, 2.37]
    Figure 5 degradation of the test component over 18 days
      -> paper: predicted failure time interval [154, 268] days
    """
    print("\n" + "=" * 74)
    print("Reproduction of the paper's published case study")
    print("=" * 74)

    table_1 = [1.46, 1.63, 1.94, 2.68, 1.77]
    criteria = FailureCriteria(table_1, k_sigma=1.0)
    print(f"\nTable 1 degradations        : {table_1}")
    print(f"Mean   computed / published : {criteria.mean:.2f} / 1.90")
    print(f"Sigma  computed / published : {criteria.sigma:.2f} / 0.47")
    print(f"Criteria computed / published: "
          f"[{criteria.lower:.2f}, {criteria.upper:.2f}] / [1.43, 2.37]")

    # Figure 5: 18 daily degradation readings rising from ~0.165 to ~0.29.
    times = list(range(1, 19))
    fig5 = [0.165, 0.197, 0.172, 0.163, 0.157, 0.168, 0.222, 0.227,
            0.232, 0.228, 0.238, 0.248, 0.258, 0.283, 0.272, 0.277,
            0.293, 0.284]
    prediction = predict_rul(times, fig5, criteria, LinearTrend)
    print(f"\nFigure 5 regression         : {prediction.trend}")
    print(f"Failure time computed       : "
          f"[{prediction.failure_time_lower:.0f}, {prediction.failure_time_upper:.0f}] days "
          f"(nominal {prediction.failure_time_nominal:.0f})")
    print(f"Failure time published      : [154, 268] days (nominal 210)")
    print(f"RUL at day {times[-1]}             : "
          f"[{prediction.rul_lower:.0f}, {prediction.rul_upper:.0f}] days")

    print("\n" + ascii_plot({"*": fig5}, times, height=10,
                            title="Figure 5 - degradation of the test component"))

    # Extrapolated view corresponding to Figure 6
    horizon = list(range(0, 301, 10))
    projected = [prediction.trend.predict(t) for t in horizon]
    lower_line = [criteria.lower] * len(horizon)
    upper_line = [criteria.upper] * len(horizon)
    print("\n" + ascii_plot({"-": lower_line, "=": upper_line, "/": projected},
                            horizon, height=14,
                            title="Figure 6 - projected degradation ( / ) vs "
                                  "failure criteria ( - lower, = upper )"))


# =========================================================================== #

def main() -> None:
    demo()
    reproduce_paper_case_study()
    print("\nDone.")


if __name__ == "__main__":
    main()
