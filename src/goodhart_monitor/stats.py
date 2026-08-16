"""Metrics and intervals. Small, boring, and separately testable on purpose.

Two rules run through this module:

  * a metric that cannot be computed returns None, never a plausible-looking
    number. A checker that reports 0.0 for "no alerts fired" invites a reader
    to treat an absence of evidence as evidence;
  * an interval is always reported beside a point estimate, and the interval is
    computed by resampling the same entities together. Resampling rows would
    treat 60 correlated hours from one patient as 60 independent facts and
    produce an interval several times too tight.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

HOLDS = "HOLDS"
FAILS = "FAILS"
INDETERMINATE = "INDETERMINATE"
NOT_APPLICABLE = "NOT_APPLICABLE"


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    """None when undefined, which is exactly when one class is absent."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def ppv(y: np.ndarray, mask: np.ndarray) -> float | None:
    """Precision among alerts. None when nothing alerted."""
    n = int(mask.sum())
    if n == 0:
        return None
    return float(y[mask].mean())


def _codes(entities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entity labels -> dense integer codes, stable across runs."""
    uniq, codes = np.unique(np.asarray(entities), return_inverse=True)
    return codes.astype(np.intp), uniq


def entity_bootstrap_ci(
    y: np.ndarray, p: np.ndarray, entities: np.ndarray,
    fn=auroc, n: int = 400, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Cluster bootstrap over entities, not rows.

    Rows within a patient are heavily correlated: consecutive hours share the
    same physiology and the same eventual outcome. Resampling rows would report
    an interval far tighter than the evidence supports, and a tight interval on
    a governance record is a claim of certainty we have not earned.
    """
    codes, uniq = _codes(entities)
    k = len(uniq)
    if k < 10:
        return None

    # Rows grouped by entity once, so a draw is a vectorised gather rather than
    # a Python loop over every entity on every iteration. With 20,000 patients
    # and 400 draws the naive version does eight million dict lookups; this
    # does none, and the arithmetic below is the standard "repeat the group
    # start, add a within-group offset" trick.
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n):
        pick = rng.integers(0, k, size=k)
        c = counts[pick]
        total = int(c.sum())
        if total == 0:
            continue
        within = np.arange(total) - np.repeat(np.concatenate([[0], np.cumsum(c)[:-1]]), c)
        idx = order[np.repeat(starts[pick], c) + within]
        v = fn(y[idx], p[idx])
        if v is not None:
            vals.append(v)
    if len(vals) < max(20, n // 10):
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def verdict_at_least(
    measured: float | None, target: float, tolerance: float,
    ci: tuple[float, float] | None,
) -> str:
    """Does `measured` reach `target`, allowing `tolerance` below it?

    INDETERMINATE is a real answer and is used whenever the interval straddles
    the line. Forcing a binary there would be the checker doing exactly what it
    accuses vendors of.
    """
    if measured is None:
        return INDETERMINATE
    floor = target - tolerance
    if ci is None:
        return HOLDS if measured >= floor else FAILS
    lo, hi = ci
    if lo >= floor:
        return HOLDS
    if hi < floor:
        return FAILS
    return INDETERMINATE


def alerts_per_100_entity_days(n_alert_rows: int, total_rows: int,
                               hours_per_row: float = 1.0) -> float | None:
    if total_rows == 0:
        return None
    days = total_rows * hours_per_row / 24.0
    if days <= 0:
        return None
    return float(100.0 * n_alert_rows / days)
