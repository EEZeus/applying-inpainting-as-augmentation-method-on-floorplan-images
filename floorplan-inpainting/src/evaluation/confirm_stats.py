"""Small, dependency-free statistics for the confirmatory study (no scipy).

Welch's t-test with the t distribution evaluated through the regularised
incomplete beta function, a Monte Carlo permutation test, Holm's step-down
adjustment, percentile intervals and Hedges' g.
"""
import math

import numpy as np


def _betacf(a, b, x, max_iter=400, eps=3e-16):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
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
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom (df may be fractional)."""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def welch(control, treatment):
    """One-sided Welch test of mean(treatment) > mean(control)."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "df": None, "p_one_sided": None, "delta": float(b.mean() - a.mean())
                if len(a) and len(b) else None}
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    delta = float(b.mean() - a.mean())
    se = math.sqrt(va + vb)
    if se == 0:
        return {"t": math.inf if delta > 0 else -math.inf, "df": None,
                "p_one_sided": 0.0 if delta > 0 else 1.0, "delta": delta}
    t = delta / se
    df = (va + vb) ** 2 / ((va ** 2) / (len(a) - 1) + (vb ** 2) / (len(b) - 1))
    return {"t": t, "df": df, "p_one_sided": t_sf(t, df), "delta": delta, "se": se}


def permutation_test(control, treatment, n_permutations=200_000, seed=20261006):
    """One-sided Monte Carlo permutation p of mean(treatment) - mean(control)."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.concatenate([a, b])
    observed = b.mean() - a.mean()
    rng = np.random.default_rng(seed)
    count, done, chunk = 0, 0, 20_000
    while done < n_permutations:
        size = min(chunk, n_permutations - done)
        keys = rng.random((size, len(pooled)))
        order = np.argsort(keys, axis=1)
        shuffled = pooled[order]
        stats = shuffled[:, len(a):].mean(1) - shuffled[:, :len(a)].mean(1)
        count += int(np.sum(stats >= observed - 1e-12))
        done += size
    return (count + 1) / (n_permutations + 1)


def holm(pvalues, alpha=0.05):
    """Holm step-down. pvalues: dict name -> p. Returns name -> (p_adj, reject)."""
    items = sorted(((p, name) for name, p in pvalues.items() if p is not None))
    m, result, running, stop = len(items), {}, 0.0, False
    for rank, (p, name) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        reject = (not stop) and p <= alpha / (m - rank)
        stop = stop or not reject
        result[name] = (running, reject)
    for name, p in pvalues.items():
        if p is None:
            result[name] = (None, False)
    return result


def hedges_g(control, treatment):
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return None
    d = (b.mean() - a.mean()) / pooled
    return float(d * (1 - 3 / (4 * (len(a) + len(b)) - 9)))


def percentile_ci(samples, level=0.95):
    samples = np.asarray(samples, float)
    lo, hi = np.percentile(samples, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)
