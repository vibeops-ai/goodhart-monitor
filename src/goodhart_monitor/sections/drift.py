"""DRIFT — windowed performance against explicit review thresholds.

Tignanelli, on what nobody sells: "having some real time model performance
benchmarks with some thresholds — hey, there's been 5% deterioration in the
AUROC for example — that triggers some sort of internal review... that's
missing." His team re-validates by hand every three months across 70+ models.

The baseline is local, and that decision is the section. Measuring each window
against the vendor's card re-reports the acceptance gap once per window and
dresses a constant up as a trend; the first version of this checker did exactly
that and lit up five of ten windows for no reason. Drift is movement away from
what this hospital actually measured on acceptance.

Two further restraints, both learned the same way:

  * a review threshold is a trigger, not a verdict. One window below the line
    on a noisy sample means look at it, and saying FAILS there would make the
    section cry wolf until the committee stops reading it. A verdict needs
    either a sustained run or a deterioration that is still present in the most
    recent window.
  * drift is a claim about time. Without a calendar the windows are cut from
    sorted entity ids, which measures sampling variation between arbitrary
    groups of patients. That is worth printing as an instrument check and is
    not worth a verdict, so an unordered stream returns INDETERMINATE and says
    what the hospital would have to supply to make it answerable.
"""
from __future__ import annotations

import numpy as np

from .. import stats
from ..card import ModelCard
from ..config import Config
from ..contract import ScoredStream


def drift(stream: ScoredStream, card: ModelCard, cfg: Config,
          baseline_auroc: float | None, baseline_ppv: float | None,
          threshold: float | None = None, ordered: bool = False) -> dict:
    thr = threshold if threshold is not None else card.threshold
    base = {
        "question": "windowed performance against explicit review thresholds",
        "baseline": "local acceptance measurement, never the vendor card",
        "thresholds": {
            "auroc_drop_from_local_baseline": cfg.drift_auroc_drop,
            "ppv_floor_fraction_of_local": cfg.drift_ppv_floor_fraction,
        },
        "baseline_auroc": None if baseline_auroc is None else round(baseline_auroc, 4),
        "baseline_row_ppv": None if baseline_ppv is None else round(baseline_ppv, 4),
    }
    if baseline_auroc is None or thr is None:
        return {**base, "verdict": stats.NOT_APPLICABLE,
                "why": "no local baseline or no threshold to monitor against"}

    df = stream.df
    entities = df["entity_id"].drop_duplicates().tolist()
    if not ordered:
        entities = sorted(entities)
    k = cfg.drift_windows
    if len(entities) < k * 2:
        return {**base, "verdict": stats.INDETERMINATE,
                "why": f"{len(entities)} entities cannot support {k} windows"}

    windows = []
    for i in range(k):
        chunk = set(entities[i * len(entities) // k:(i + 1) * len(entities) // k])
        w = df[df.entity_id.isin(chunk)]
        wy, wp = w["label"].to_numpy(), w["score"].to_numpy()
        wmask = wp >= thr
        w_auroc = stats.auroc(wy, wp)
        w_ppv = stats.ppv(wy, wmask)

        review_reasons = []
        if w_auroc is not None and w_auroc < baseline_auroc - cfg.drift_auroc_drop:
            review_reasons.append("auroc")
        if (baseline_ppv is not None and w_ppv is not None
                and w_ppv < baseline_ppv * cfg.drift_ppv_floor_fraction):
            review_reasons.append("ppv")
        undetermined = w_auroc is None or int(wmask.sum()) < cfg.min_cell

        windows.append({
            "window": i + 1,
            "entities": len(chunk),
            "rows": int(len(w)),
            "auroc": None if w_auroc is None else round(w_auroc, 4),
            "row_ppv": None if w_ppv is None else round(w_ppv, 4),
            "alerts": int(wmask.sum()),
            "review": bool(review_reasons) and not undetermined,
            "review_reasons": review_reasons,
            "underpowered": bool(undetermined),
        })

    flags = [w["review"] for w in windows]
    n_review = sum(flags)
    longest_run, run = 0, 0
    for f in flags:
        run = run + 1 if f else 0
        longest_run = max(longest_run, run)
    latest = bool(flags[-1])

    aurocs = [w["auroc"] for w in windows if w["auroc"] is not None]
    out = {
        **base,
        "ordering": ("stream order as supplied, treated as calendar order" if ordered
                     else "constructed from sorted entity id; the stream carries "
                          "no calendar"),
        "windows": windows,
        "windows_triggering_review": n_review,
        "longest_consecutive_run": longest_run,
        "latest_window_triggers_review": latest,
        "run_length_for_verdict": cfg.drift_review_run_for_fail,
        "auroc_spread": (round(float(max(aurocs) - min(aurocs)), 4)
                         if len(aurocs) > 1 else None),
    }

    if not ordered:
        return {**out, "verdict": stats.INDETERMINATE,
                "why": "drift is a claim about time and this stream carries none. "
                       "The windows above are cut from sorted entity ids, so the "
                       "spread between them measures sampling variation between "
                       "arbitrary groups of patients, not deterioration. To make "
                       "this section answerable, supply the stream in calendar "
                       "order and pass --ordered",
                "note": "reported as an instrument check: it shows the review "
                        "thresholds computing against a local baseline, on real "
                        "scores, and says nothing about this deployment over time"}

    if longest_run >= cfg.drift_review_run_for_fail or latest:
        why = ("the most recent window is below a review threshold"
               if latest else
               f"{longest_run} consecutive windows below a review threshold")
        return {**out, "verdict": stats.FAILS, "why": why}
    if n_review:
        return {**out, "verdict": stats.INDETERMINATE,
                "why": f"{n_review} isolated window(s) crossed a review threshold "
                       "and recovered. That is a trigger for review, not a "
                       "finding, and this record does not know the outcome of "
                       "that review"}
    return {**out, "verdict": stats.HOLDS,
            "why": "no window crossed a review threshold against the local baseline"}
