"""ACCEPTANCE — does the card's headline number hold on this population?

The committee's first question about any vendor number: does it hold on our
population, measured with our data rather than the vendor's.

The whole section is one comparison, but the comparison is the product. A card
number is measured on the vendor's population; this one is measured on yours,
with a cluster bootstrap so the gap is reported with an interval rather than as
a bare point that invites over-reading.
"""
from __future__ import annotations

from .. import stats
from ..card import ModelCard
from ..config import Config
from ..contract import ScoredStream


def acceptance(stream: ScoredStream, card: ModelCard, cfg: Config) -> dict:
    claim = card.of_kind("auroc")
    base = {
        "question": "does the card's headline number hold on this population?",
        "n_rows": stream.n_rows,
        "n_entities": stream.n_entities,
    }
    if claim is None or claim.value is None:
        return {**base,
                "verdict": stats.NOT_APPLICABLE,
                "why": "the card states no discrimination number to test. An "
                       "unfalsifiable card is itself a finding for the committee"}

    y, p = stream.y, stream.p
    ent = stream.df["entity_id"].to_numpy()
    measured = stats.auroc(y, p)
    ci = stats.entity_bootstrap_ci(
        y, p, ent, fn=stats.auroc, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed)

    floor = claim.value - cfg.auroc_tolerance
    verdict = stats.verdict_at_least(measured, claim.value, cfg.auroc_tolerance, ci)
    if measured is None:
        why = "one outcome class only, so discrimination is undefined here"
    elif verdict == stats.FAILS:
        why = (f"measured {measured:.4f} against a floor of {floor:.4f} "
               f"(card {claim.value:g} less {cfg.auroc_tolerance:g} tolerance)"
               + (f", and the interval's upper bound {ci[1]:.4f} is still below it"
                  if ci else ""))
    elif verdict == stats.INDETERMINATE:
        why = (f"measured {measured:.4f} but the interval "
               f"[{ci[0]:.4f}, {ci[1]:.4f}] straddles the {floor:.4f} floor, so "
               f"this population cannot settle the claim either way"
               if ci else "no interval could be computed on this population")
    else:
        why = (f"measured {measured:.4f} reaches the {floor:.4f} floor"
               + (f"; the interval's lower bound is {ci[0]:.4f}" if ci else ""))

    return {
        **base,
        "claim_id": claim.id,
        "card_claim": claim.text,
        "card_value": claim.value,
        "card_population": claim.population,
        "measured_auroc": None if measured is None else round(measured, 4),
        "ci95_entity_bootstrap": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "gap_vs_card": None if measured is None else round(claim.value - measured, 4),
        "tolerance": cfg.auroc_tolerance,
        "verdict": verdict,
        "why": why,
        "note": "interval is a cluster bootstrap over entities, not rows: hours "
                "within one stay are not independent observations",
    }
