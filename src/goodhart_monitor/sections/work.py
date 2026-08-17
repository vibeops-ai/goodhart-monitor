"""WORK — what work does the alert stream create, and is that work valuable?

The question no vendor card answers: what work does this model create, and is
that work valuable enough to justify the burden?

Two counting decisions here are the difference between an honest section and a
flattering one, and both were wrong in the first version of this checker:

  * A catch is a first alert that fires BEFORE onset. An alert on a patient who
    is already septic is work, not a catch. Counting it as a catch improved the
    headline ratio by a third.
  * The burden ratio is never reported alone. A threshold can buy an excellent
    patients-per-catch number by refusing to alert at all, so sensitivity sits
    beside it on the same line and the record says why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import stats
from ..card import ModelCard
from ..config import Config
from ..contract import ScoredStream


def entity_table(stream: ScoredStream, threshold: float) -> pd.DataFrame:
    """Collapse rows to what governance counts: entities, alerts, timing."""
    df = stream.df
    fired = df["score"] >= threshold
    g = df.assign(_fired=fired).groupby("entity_id", sort=True)

    out = pd.DataFrame({
        "positive": g["label"].max().astype(bool),
        "n_rows": g["label"].size(),
        "n_alert_rows": g["_fired"].sum().astype(int),
    })
    out["alerted"] = out["n_alert_rows"] > 0

    if stream.has_time:
        first_alert = (df[fired].groupby("entity_id")["t"].min()
                       .rename("first_alert"))
        out = out.join(first_alert)
    else:
        out["first_alert"] = np.nan

    if stream.has_onset:
        onset = df.groupby("entity_id")["onset_t"].min().rename("onset")
        out = out.join(onset)
    else:
        out["onset"] = np.nan

    return out.reset_index()


def work(stream: ScoredStream, card: ModelCard, cfg: Config,
         threshold: float | None = None) -> dict:
    thr = threshold if threshold is not None else card.threshold
    base = {"question": "what work does the alert stream create at the shipped threshold?"}
    if thr is None:
        return {**base, "verdict": stats.NOT_APPLICABLE,
                "why": "the card names no shipped threshold, so there is no alert "
                       "stream to count. Ask the vendor which threshold you are "
                       "running; that question alone is often informative"}

    y, p = stream.y, stream.p
    mask = p >= thr
    et = entity_table(stream, thr)
    n_ent = len(et)
    alerted = et[et.alerted]
    positives = et[et.positive]

    hour_ppv = stats.ppv(y, mask)
    ent_ppv = None if len(alerted) == 0 else float(alerted.positive.mean())

    # a catch is a first alert strictly before onset; anything later is work
    if stream.has_time and stream.has_onset:
        catchable = alerted.dropna(subset=["first_alert", "onset"])
        actionable = catchable[catchable.first_alert < catchable.onset]
        n_actionable = int(len(actionable))
        nne = round(len(alerted) / n_actionable, 1) if n_actionable else None
        sens_actionable = (round(n_actionable / len(positives), 4)
                           if len(positives) else None)
    else:
        n_actionable, nne, sens_actionable = None, None, None

    burden = ("" if nne is None else
              f" Staffing consequence: {nne} entities evaluated per actionable "
              f"catch, at {sens_actionable:.1%} sensitivity.")

    claim = card.of_kind("ppv")
    if claim is not None and claim.value is not None and hour_ppv is not None:
        floor = claim.value * cfg.ppv_tolerance_fraction
        verdict = stats.HOLDS if hour_ppv >= floor else stats.FAILS
        why = (f"alert precision {hour_ppv:.4f} against a floor of {floor:.4f} "
               f"({cfg.ppv_tolerance_fraction:g} of the card's {claim.value:g})."
               + burden)
        if int(mask.sum()) < cfg.min_cell:
            verdict = stats.INDETERMINATE
            why = (f"only {int(mask.sum())} alert rows fired, below the "
                   f"{cfg.min_cell} needed to quote a precision." + burden)
    elif hour_ppv is None:
        verdict, why = stats.NOT_APPLICABLE, (
            "nothing alerted at this threshold, so there is no alert stream to "
            "count. That is a finding about the threshold, not about the model")
    else:
        verdict, why = stats.NOT_APPLICABLE, (
            "the card claims no alert precision, so there is nothing to test "
            "against. The burden numbers below stand on their own." + burden)

    return {
        **base,
        "claim_id": None if claim is None else claim.id,
        "card_claim": None if claim is None else claim.text,
        "card_value": None if claim is None else claim.value,
        "threshold": thr,
        "n_alert_rows": int(mask.sum()),
        "alerts_per_100_entity_days": None if not stream.has_time else
        (lambda v: None if v is None else round(v, 1))(
            stats.alerts_per_100_entity_days(int(mask.sum()), stream.n_rows)),
        "share_of_entities_ever_alerted": round(len(alerted) / n_ent, 4) if n_ent else None,
        "row_level_ppv": None if hour_ppv is None else round(hour_ppv, 4),
        "entity_level_ppv": None if ent_ppv is None else round(ent_ppv, 4),
        "actionable_catches": n_actionable,
        "entities_evaluated_per_actionable_catch": nne,
        "sensitivity_actionable": sens_actionable,
        "tolerance_fraction": cfg.ppv_tolerance_fraction,
        "verdict": verdict,
        "why": why,
        "note": "entities evaluated per actionable catch is the operational "
                "unit; the Epic sepsis model needed 8 at Michigan (Wong et al., "
                "JAMA Intern Med 2021). Only a first alert "
                "before onset counts as a catch. Read it beside the sensitivity: "
                "a threshold can buy a flattering ratio by refusing to alert, and "
                "no card mentions that trade",
    }
