"""TIMING — does it warn before the event, or notice it afterwards?

This is the claim class the Epic sepsis model card got wrong. External
validation found a tool that was good at case finding, identifying patients
after the fact, while the card described prediction before onset (Wong et al.,
External Validation of a Widely Implemented Proprietary Sepsis Prediction
Model, JAMA Intern Med 2021).

The section reports the long leads and the short ones separately on purpose. A
median lead of forty hours sounds magnificent and is often an early-stay alert
on a patient who deteriorates two days later; the bedside experiences that as
an unexplained alarm, not a warning. The actionable window makes the difference
visible instead of averaging it away.
"""
from __future__ import annotations

import numpy as np

from .. import stats
from ..card import ModelCard
from ..config import Config
from ..contract import ScoredStream
from .work import entity_table


def timing(stream: ScoredStream, card: ModelCard, cfg: Config,
           threshold: float | None = None) -> dict:
    thr = threshold if threshold is not None else card.threshold
    claim = card.of_kind("lead_time")
    base = {"question": "does it warn before the event, or notice it afterwards?",
            "card_claim": None if claim is None else claim.text}

    if not stream.has_time:
        return {**base, "verdict": stats.NOT_APPLICABLE,
                "why": "the stream has no time column, so 'before' has no meaning "
                       "here. This is the right answer for an entity-level outcome "
                       "such as readmission, and it is reported rather than skipped"}
    if not stream.has_onset:
        return {**base, "verdict": stats.NOT_APPLICABLE,
                "why": "no onset time is available and the label does not vary "
                       "within an entity, so lead time cannot be measured"}
    if thr is None:
        return {**base, "verdict": stats.NOT_APPLICABLE,
                "why": "no shipped threshold, so there are no alerts to time"}

    et = entity_table(stream, thr)
    positives = et[et.positive]
    caught = positives[positives.alerted].dropna(subset=["first_alert", "onset"])
    if len(positives) == 0:
        return {**base, "verdict": stats.INDETERMINATE,
                "why": "no positive outcomes in the window"}
    if len(caught) == 0:
        return {**base, "positives": int(len(positives)), "caught_at_all": 0,
                "verdict": stats.FAILS,
                "why": "the model alerted on no positive entity at this threshold"}

    lead = (caught.onset - caught.first_alert).to_numpy(dtype=float)
    # A median lead time is the single most misleading number in this category:
    # it averages a two-day-early alert together with a one-hour-early one and
    # reports something no clinician experiences. The distribution is what the
    # committee needs, so it ships on the record rather than in an appendix.
    edges = [-np.inf, 0.0, 6.0, 12.0, 24.0, 48.0, np.inf]
    names = ["after onset", "0-6h", "6-12h", "12-24h", "24-48h", "48h+"]
    hist = [{"bucket": nm,
             "catches": int(((lead > lo) & (lead <= hi)).sum())}
            for nm, lo, hi in zip(names, edges[:-1], edges[1:])]
    before = lead > 0
    w = cfg.actionable_window_hours
    in_window = before & (lead <= w)
    share_after = float(1.0 - before.mean())

    verdict = (stats.HOLDS if before.mean() >= cfg.min_share_before_onset
               else stats.FAILS)
    why = (f"{int(before.sum())} of {len(caught)} catches ({before.mean():.1%}) "
           f"preceded onset, against a policy floor of "
           f"{cfg.min_share_before_onset:.0%}. Separately, {int(in_window.sum())} "
           f"({in_window.mean():.1%}) landed inside the {w:g}h actionable window; "
           f"the rest were early enough that the bedside may not connect the "
           f"alert to the deterioration")
    if len(caught) < cfg.min_cell:
        verdict = stats.INDETERMINATE
        why = (f"only {len(caught)} positive entities were alerted on, below the "
               f"{cfg.min_cell} needed to quote a share")

    return {
        **base,
        "claim_id": None if claim is None else claim.id,
        "positives": int(len(positives)),
        "caught_at_all": int(len(caught)),
        "sensitivity_any_alert": round(len(caught) / len(positives), 4),
        "caught_before_onset": int(before.sum()),
        "share_of_catches_after_onset": round(share_after, 4),
        "median_lead_hours_when_early": (round(float(np.median(lead[before])), 1)
                                         if before.any() else None),
        "lead_time_distribution": hist,
        "actionable_window_hours": w,
        "caught_within_window": int(in_window.sum()),
        "share_of_catches_within_window": round(float(in_window.mean()), 4),
        "min_share_before_onset": cfg.min_share_before_onset,
        "verdict": verdict,
        "why": why,
        "note": "an alert that fires after onset is case finding, not prediction. "
                "Long leads are reported beside the actionable window because a "
                "warning two days early is experienced at the bedside as an "
                "unexplained alarm, and averaging the two hides that",
    }
