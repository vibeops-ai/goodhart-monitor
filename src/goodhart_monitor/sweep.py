"""Every verdict this record could have had, at every threshold it could run at.

A verification record answers "is this deployment acceptable at the threshold
you shipped." The question a committee asks thirty seconds later is "what
threshold should we be running", and that question has no honest answer from a
single point: precision, alert volume, sensitivity and lead time all move
together, and moving one to look good moves another to look bad.

So this sweeps the threshold across its whole range and reports the measured
quantities at each step. It states no verdicts. Verdicts come from policy, and
policy belongs to the hospital; anything reading this artifact applies its own
thresholds to these measurements. The lead-time counts are reported in fine
bins rather than as a single "within the window" count for the same reason:
the window is policy, so the reader must be able to change it without asking
us to recompute anything.

The output is small enough to hand to a browser, which is the point. A number
that only exists inside our pipeline is a number the buyer has to trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .card import ModelCard
from .config import Config
from .contract import ScoredStream

# Lead-time bin edges in hours. A reader may set an actionable window to any of
# these edges and get an exact count by summing bins; offering a free-text
# window would mean interpolating, and an interpolated clinical number on a
# governance artifact is a fabricated one.
LEAD_EDGES = [0.0, 3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0]
SELECTABLE_WINDOWS = [6.0, 12.0, 24.0, 48.0]


def _grid(p: np.ndarray, n: int, shipped: float | None = None) -> list[float]:
    """Threshold candidates, drawn from the score distribution itself.

    A linear grid over [0, 1] spends most of its points where no patient ever
    scores. Quantiles put the resolution where the operating points are.

    The shipped threshold is always included, exactly. Showing a reader the
    nearest grid point to the threshold they are actually running, and labelling
    it as the one they are running, is a small lie of the kind this whole
    exercise is about.
    """
    qs = list(np.quantile(p, np.linspace(0.50, 0.99995, n)))
    if shipped is not None:
        qs.append(float(shipped))
    return [float(v) for v in np.unique(np.round(qs, 9))]


def _staircase(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per entity, the points in time where its running maximum score rises.

    Only these points can ever be a first alert: if the running max has not
    risen, a lower threshold would already have fired earlier. Collapsing to the
    staircase turns "when does this entity first alert at threshold t" into a
    single sorted sweep instead of one search per entity per threshold.

    Returns (entity_code, step_value, step_time), unsorted.
    """
    d = df.sort_values(["entity_id", "t"], kind="mergesort")
    codes, _ = pd.factorize(d["entity_id"], sort=True)
    score = d["score"].to_numpy()

    # running max within each entity. np.maximum.accumulate does not reset at
    # group boundaries, so each entity's slice is accumulated in place.
    boundary = np.r_[True, codes[1:] != codes[:-1]]
    run_max = score.copy()
    starts = np.flatnonzero(boundary)
    for a, b in zip(starts, np.r_[starts[1:], len(codes)]):
        np.maximum.accumulate(run_max[a:b], out=run_max[a:b])

    # a step is the first row of an entity, or a row where the running max rose
    rises = boundary.copy()
    rises[1:] |= (run_max[1:] > run_max[:-1]) & ~boundary[1:]
    return codes[rises], run_max[rises], d["t"].to_numpy(dtype=float)[rises]


def sweep(stream: ScoredStream, card: ModelCard, cfg: Config,
          n_points: int = 220) -> dict:
    df = stream.df
    y, p = stream.y, stream.p
    ent = df["entity_id"].to_numpy()
    thresholds = _grid(p, n_points, card.threshold)

    # Row-level counts come from one sort of the scores: the number of rows at
    # or above a threshold, and how many of them were true, are both prefix
    # sums over scores in descending order.
    order = np.argsort(-p, kind="stable")
    p_desc = p[order]
    cum_true = np.cumsum(y[order])

    n_rows = len(y)
    ent_codes, uniq_ent = pd.factorize(pd.Series(ent), sort=True)
    n_ent = len(uniq_ent)
    ent_pos = np.zeros(n_ent, dtype=bool)
    np.maximum.at(ent_pos, ent_codes, y.astype(bool))
    n_pos_ent = int(ent_pos.sum())
    days = n_rows / 24.0

    has_timing = stream.has_time and stream.has_onset
    if has_timing:
        step_e, step_v, step_t = _staircase(df)
        # process steps from the highest score down, so a falling threshold
        # only ever moves an entity's first alert earlier
        s_order = np.argsort(-step_v, kind="stable")
        step_e, step_v, step_t = step_e[s_order], step_v[s_order], step_t[s_order]
        first_alert = np.full(n_ent, np.inf)
        cursor = 0

        onset = np.full(n_ent, np.nan)
        o = df.groupby("entity_id", sort=True)["onset_t"].min()
        onset[:] = o.to_numpy(dtype=float)

    points = []
    for thr in sorted(thresholds, reverse=True):
        k = int(np.searchsorted(-p_desc, -thr, side="right"))
        n_true = int(cum_true[k - 1]) if k > 0 else 0
        ppv = (n_true / k) if k else None

        row = {
            "threshold": thr,
            "n_alert_rows": k,
            "row_ppv": None if ppv is None else round(ppv, 6),
            "alerts_per_100_entity_days": round(100.0 * k / days, 3) if days else None,
        }

        if has_timing:
            while cursor < len(step_v) and step_v[cursor] >= thr:
                e = step_e[cursor]
                if step_t[cursor] < first_alert[e]:
                    first_alert[e] = step_t[cursor]
                cursor += 1

            alerted = np.isfinite(first_alert)
            caught = alerted & ent_pos & ~np.isnan(onset)
            lead = onset[caught] - first_alert[caught]
            early = lead[lead > 0]

            n_before = int(early.size)
            row["entities_alerted"] = int(alerted.sum())
            row["caught_at_all"] = int(caught.sum())
            row["caught_before_onset"] = n_before
            row["actionable_catches"] = n_before
            row["sensitivity_actionable"] = (round(n_before / n_pos_ent, 6)
                                             if n_pos_ent else None)
            row["entities_per_actionable_catch"] = (
                round(float(alerted.sum()) / n_before, 3) if n_before else None)
            row["lead_bins"] = [int(((early > lo) & (early <= hi)).sum())
                                for lo, hi in zip(LEAD_EDGES[:-1], LEAD_EDGES[1:])]
            row["lead_beyond_last_bin"] = int((early > LEAD_EDGES[-1]).sum())
            row["median_lead_when_early"] = (round(float(np.median(early)), 2)
                                             if early.size else None)
        points.append(row)

    points.reverse()          # back to ascending threshold, the reading order

    auroc = stats.auroc(y, p)
    ci = stats.entity_bootstrap_ci(y, p, ent, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed)

    return {
        "kind": "goodhart.monitor.sweep/1",
        "record": cfg.record_id,
        "subject": card.as_dict(),
        "stream": {"rows": n_rows, "entities": n_ent, "positive_entities": n_pos_ent,
                   "entity_days": round(days, 2), "has_timing": has_timing},
        # discrimination does not depend on the threshold, which is itself the
        # point worth making: you cannot move the slider to fix a card claim
        "auroc": None if auroc is None else round(auroc, 6),
        "auroc_ci95": None if ci is None else [round(ci[0], 6), round(ci[1], 6)],
        "lead_edges": LEAD_EDGES,
        "selectable_windows": SELECTABLE_WINDOWS,
        "shipped_threshold": card.threshold,
        "points": points,
        "note": "measurements only. This artifact states no verdicts, because a "
                "verdict is a measurement plus a policy and the policy is the "
                "hospital's",
    }
