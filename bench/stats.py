"""Small-sample statistics for the pooled report, estimates and coverage.

The bench compares harnesses on 10-20 paired tasks, far too few for normal
approximations or a plain percentile bootstrap to be trusted on their own,
so every interval here is chosen for that size: a t interval on per-task log
ratios, Wilson score intervals, Newcombe's paired interval for pass rates and
an exact McNemar test. One module serves the REPORT tab and the run-design
assistant, so both quote the same numbers.

Pure stdlib and deterministic: t quantiles come from the incomplete beta
function rather than a table, and every random draw is seeded through
stable_seed, never Python's per-process salted hash().
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

# 2: trials whose usage was not recorded left the token and cost figures (same ledger, new numbers).
# 3: verdicts are checked against trials cut short or unrecorded costing as much as their harness's
#    most expensive trial, and a floor below the harness's uncut trials on its task is set aside.
STATS_VERSION = 3

T = TypeVar("T")

_DEFAULT_TOLERANCE = 1.5  # results._verdict's reading of a single published value


def _check_level(level: float) -> float:
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1 (got {level!r})")
    return float(level)


# Distributions --------------------------------------------------------------------

def z_quantile(level: float) -> float:
    """Two-sided normal quantile: 1.96 for 0.95, 1.645 for 0.90."""
    return statistics.NormalDist().inv_cdf(0.5 + _check_level(level) / 2.0)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    """Student t cumulative distribution function."""
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isnan(x):
        raise ValueError("x is not a number")
    if math.isinf(x):
        return 1.0 if x > 0 else 0.0
    if x == 0.0:
        return 0.5
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - tail if x > 0 else tail


@lru_cache(maxsize=8192)
def _t_quantile(level: float, df: float) -> float:
    target = 0.5 + level / 2.0
    lo, hi = 0.0, max(2.0, z_quantile(level) * 2.0)
    while t_cdf(hi, df) < target:
        lo, hi = hi, hi * 2.0
        if hi > 1e12:
            break
    while hi - lo > 1e-9 * max(1.0, hi):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def t_quantile(level: float, df: float) -> float:
    """Two-sided t quantile: 2.262 for level 0.95 at df 9, 1.833 for 0.90."""
    level = _check_level(level)
    if isinstance(df, bool) or not isinstance(df, (int, float)) or df < 1:
        raise ValueError(f"df must be at least 1 (got {df!r})")
    return _t_quantile(level, float(df))


def t_crit(level: float, df: float) -> float:
    """The planner's name for t_quantile."""
    return t_quantile(level, df)


# Intervals -------------------------------------------------------------------------

def wilson(successes: float, n: float, level: float) -> tuple[float, float] | None:
    """Wilson score interval for a proportion. Fractional successes are
    allowed (a task-weighted pass rate). None when n is not positive."""
    if n is None or n <= 0:
        return None
    z = z_quantile(level)
    p = min(max(successes / n, 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0)) / denom
    # Rounding leaves 1e-17 where the bound is exactly 0 or 1.
    lo = 0.0 if p == 0.0 else max(0.0, center - half)
    hi = 1.0 if p == 1.0 else min(1.0, center + half)
    return lo, hi


def log_ratio_summary(log_ratios: Sequence[float], level: float) -> dict[str, Any]:
    """Geometric mean of per-task ratios with a t interval on the log scale.
    lo and hi are None below two values."""
    values = [float(v) for v in log_ratios]
    n = len(values)
    if n == 0:
        raise ValueError("log_ratio_summary needs at least one value")
    mean = math.fsum(values) / n
    out: dict[str, Any] = {"estimate": math.exp(mean), "lo": None, "hi": None, "sd_log": None, "gsd": None, "n": n}
    if n >= 2:
        sd = statistics.stdev(values)
        half = t_quantile(level, n - 1) * sd / math.sqrt(n)
        out.update(lo=math.exp(mean - half), hi=math.exp(mean + half), sd_log=sd, gsd=math.exp(sd))
    return out


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = math.floor(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def bootstrap_ci(values: Sequence[T], stat: Callable[[Sequence[T]], float], *, resamples: int, seed: int,
                 level: float) -> tuple[float, float] | None:
    """Percentile bootstrap over whole items (tasks). None below five items,
    where a percentile interval is not worth showing."""
    level = _check_level(level)
    items = list(values)
    n = len(items)
    if n < 5 or resamples < 1:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(int(resamples)):
        try:
            value = float(stat(rng.choices(items, k=n)))
        except (ZeroDivisionError, ValueError, statistics.StatisticsError):
            continue
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return None
    draws.sort()
    alpha = (1.0 - level) / 2.0
    return _percentile(draws, alpha), _percentile(draws, 1.0 - alpha)


def newcombe_paired(a: int, b: int, c: int, d: int, level: float) -> tuple[float, float]:
    """Newcombe (1998) method 10 interval for p_arm - p_base on paired data.

    a: both passed; b: only the arm passed; c: only the baseline passed;
    d: neither. The phi coefficient carries Newcombe's correction (zero when
    0 <= ad - bc <= n/2), which keeps the interval from collapsing to a
    point when there are no discordant tasks.
    """
    counts = (a, b, c, d)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts):
        raise ValueError("newcombe_paired takes four non-negative counts")
    n = a + b + c + d
    if n == 0:
        raise ValueError("newcombe_paired needs at least one pair")
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, level)  # type: ignore[misc]
    l2, u2 = wilson(a + c, n, level)  # type: ignore[misc]
    margins = (a + b) * (c + d) * (a + c) * (b + d)
    if margins == 0:
        phi = 0.0
    else:
        cross = a * d - b * c
        if cross > n / 2.0:
            phi = (cross - n / 2.0) / math.sqrt(margins)
        elif cross >= 0:
            phi = 0.0
        else:
            phi = cross / math.sqrt(margins)
    diff = p1 - p2
    below = math.sqrt(max((p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    above = math.sqrt(max((u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return max(-1.0, diff - below), min(1.0, diff + above)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p value: a binomial test on the discordant pairs."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def leave_one_out(values: Sequence[float], stat: Callable[[Sequence[float]], float]) -> tuple[float, float] | None:
    """(min, max) of the statistic with each item left out in turn."""
    items = list(values)
    if len(items) < 2:
        return None
    results = [float(stat(items[:i] + items[i + 1:])) for i in range(len(items))]
    return min(results), max(results)


# Sample sizes ---------------------------------------------------------------------

def n_needed_ratio(sd_log: float, halfwidth_factor: float, level: float, n_max: int = 5000) -> int:
    """Smallest n >= 2 whose t interval on log ratios is within x/÷ halfwidth_factor:
    t(level, n-1) * sd / sqrt(n) <= ln(halfwidth_factor)."""
    if halfwidth_factor <= 1.0:
        raise ValueError("halfwidth_factor must be above 1, e.g. 1.25 for +/-25%")
    if sd_log < 0 or not math.isfinite(sd_log):
        raise ValueError("sd_log must be a finite number, zero or more")
    if sd_log == 0:
        return 2
    target = math.log(halfwidth_factor)
    # t > z for every df, so the normal answer is a lower bound to search up from.
    n = max(2, math.ceil((z_quantile(level) * sd_log / target) ** 2) - 2)
    while n < n_max:
        if t_quantile(level, n - 1) * sd_log / math.sqrt(n) <= target:
            return n
        n += 1
    return n_max


def required_pairs(sd_log: float, halfwidth: float, level: float = 0.90) -> int:
    """Pairs for a multiple known to +/-halfwidth (0.25 = +/-25%):
    22 at sd 0.6 and 11 at sd 0.4, at 90%."""
    return n_needed_ratio(sd_log, 1.0 + halfwidth, level)


def mde_pp(sd_d: float, n: int, level: float) -> float:
    """Smallest pass-rate difference (percentage points) n paired tasks can see."""
    if n < 1:
        raise ValueError("mde_pp needs at least one pair")
    return 100.0 * z_quantile(level) * sd_d / math.sqrt(n)


def n_needed_pass(sd_d: float, halfwidth_pp: float, level: float) -> int:
    """Paired tasks for a pass-rate difference known to +/-halfwidth_pp points."""
    if halfwidth_pp <= 0:
        raise ValueError("halfwidth_pp must be positive")
    raw = (z_quantile(level) * sd_d / (halfwidth_pp / 100.0)) ** 2
    return max(1, math.ceil(raw - 1e-9))


# Helpers ---------------------------------------------------------------------------

def stable_seed(*parts: object) -> int:
    """A seed that is the same in every process for the same parts."""
    text = "|".join(map(str, parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def quartiles(values: Sequence[float]) -> dict[str, float] | None:
    """{min, q1, median, q3, max}; None for no values."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return {"min": ordered[0], "q1": _percentile(ordered, 0.25), "median": _percentile(ordered, 0.5),
            "q3": _percentile(ordered, 0.75), "max": ordered[-1]}


def pooled_sd(entries: Sequence[tuple[float | None, int]]) -> float | None:
    """Degrees-of-freedom weighted sd over (sd_log, n) entries with n >= 3."""
    num = den = 0.0
    for sd, n in entries:
        if sd is None or n is None or n < 3 or not math.isfinite(sd):
            continue
        num += (n - 1) * sd * sd
        den += n - 1
    return math.sqrt(num / den) if den else None


def reference_band(ref: Mapping[str, Any] | None, tolerance: float = _DEFAULT_TOLERANCE) -> tuple[float, float] | None:
    """A published figure as a band: [low, high], or [v/tol, v*tol] for one value."""
    if not isinstance(ref, Mapping):
        return None
    try:
        if ref.get("value") is not None:
            value = float(ref["value"])
            return value / tolerance, value * tolerance
        if ref.get("low") is not None and ref.get("high") is not None:
            return float(ref["low"]), float(ref["high"])
    except (TypeError, ValueError):
        return None
    return None


def band_verdict(ci: Sequence[float | None] | None, ref: Mapping[str, Any] | None,
                 tolerance: float | None = None) -> str | None:
    """Where an interval sits against a published figure, with the band rules
    of results._verdict: 'in line', 'lower', 'higher' or 'undecided'."""
    if not ci or len(ci) != 2 or ci[0] is None or ci[1] is None:
        return None
    if tolerance is None:
        tolerance = _DEFAULT_TOLERANCE
        if isinstance(ref, Mapping):
            try:
                given = float(ref.get("tolerance"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                given = 0.0
            if given >= 1.0:
                tolerance = given
    band = reference_band(ref, tolerance)
    if band is None:
        return None
    lo, hi = float(ci[0]), float(ci[1])
    if lo >= band[0] and hi <= band[1]:
        return "in line"
    if hi < band[0]:
        return "lower"
    if lo > band[1]:
        return "higher"
    return "undecided"


def lpt_makespan(durations: Sequence[float], slots: int) -> float:
    """Wall time of jobs on `slots` parallel workers, longest first (LPT)."""
    return _schedule(sorted((float(d) for d in durations), reverse=True), slots)


def list_schedule(durations_in_order: Sequence[float], slots: int) -> float:
    """Wall time when jobs start in the given order on the first free worker."""
    return _schedule([float(d) for d in durations_in_order], slots)


def _schedule(durations: list[float], slots: int) -> float:
    if not durations:
        return 0.0
    loads = [0.0] * max(1, int(slots))
    heapq.heapify(loads)
    for d in durations:
        heapq.heappush(loads, heapq.heappop(loads) + max(d, 0.0))
    return max(loads)


@dataclass(frozen=True)
class RatioEstimate:
    n: int
    multiple: float
    ci: tuple[float, float] | None
    mean_log: float
    sd_log: float | None
    sd_source: str      # observed | pooled (or the caller's label) | prior


def paired_log_ratio(base: Mapping[Any, float], arm: Mapping[Any, float], pooled_sd: float | None = None,
                     prior_sd: float = 0.6, *, level: float = 0.90, min_observed: int = 5,
                     pooled_source: str = "pooled") -> RatioEstimate | None:
    """Baseline ÷ arm multiple over the keys both sides have (values > 0).

    Below min_observed pairs the observed spread is too noisy to size an
    interval with, so a pooled spread from other comparisons (or the prior)
    is used instead, and sd_source says which.
    """
    keys = [k for k in base if k in arm and (base[k] or 0) > 0 and (arm[k] or 0) > 0]
    if not keys:
        return None
    d = [math.log(base[k]) - math.log(arm[k]) for k in keys]
    n = len(d)
    mean = math.fsum(d) / n
    if n >= min_observed:
        sd: float | None = statistics.stdev(d)
        source = "observed"
    elif pooled_sd is not None:
        sd, source = pooled_sd, pooled_source
    else:
        sd, source = prior_sd, "prior"
    ci = None
    if n >= 2 and sd is not None:
        half = t_quantile(level, n - 1) * sd / math.sqrt(n)
        ci = (math.exp(mean - half), math.exp(mean + half))
    return RatioEstimate(n=n, multiple=math.exp(mean), ci=ci, mean_log=mean, sd_log=sd, sd_source=source)
