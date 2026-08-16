"""Assemble, canonicalise and hash a verification record.

The record is the deliverable, so it obeys the rules the rest of the company
obeys: no PASS in the vocabulary, LIMITS carried at the same weight as the
findings, and content addressing so the same inputs give the same bytes years
later. Nothing that varies between machines or runs goes into the hash: no
wall clock, no paths, no library versions, no hostname. A record that changes
because it was produced on a different laptop cannot be re-run in a deposition.
"""
from __future__ import annotations

import hashlib
import json

from . import stats
from .card import ModelCard
from .config import Config
from .contract import ScoredStream
from .sections import acceptance, work, timing, drift, subgroups

SCHEMA = "goodhart.monitor/1"

DEFAULT_LIMITS = [
    "outcome labels are whatever the supplied stream calls adjudicated; the "
    "checker does not re-adjudicate them and cannot detect a mislabelled outcome",
    "verification covers the population, threshold and window in this stream. It "
    "says nothing about any other population, threshold or window",
    "no PASS verdict exists. NO finding within scope is not the same claim as safe",
]


def stable(obj):
    """Normalise numbers so the hash does not depend on the language reading it.

    Python writes 12.0 where JavaScript writes 12, and a record whose hash
    verifies in Python but not in the browser that displays it is not
    content-addressed in any useful sense. Integral floats become ints once,
    here, before the bytes are written or hashed.
    """
    if isinstance(obj, dict):
        return {k: stable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [stable(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def canonical(obj) -> str:
    return json.dumps(stable(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


def headline(sections: dict) -> dict:
    """What a committee reads first: the verdict tally, honestly bucketed."""
    tally: dict[str, int] = {}
    for name in ("acceptance", "work", "timing", "drift"):
        v = sections.get(name, {}).get("verdict")
        if v:
            tally[v] = tally.get(v, 0) + 1
    failing = [n for n in ("acceptance", "work", "timing", "drift")
               if sections.get(n, {}).get("verdict") == stats.FAILS]
    return {
        "tally": tally,
        "sections_failing": failing,
        "overall": (stats.FAILS if failing else
                    stats.INDETERMINATE if tally.get(stats.INDETERMINATE) else
                    "NO_FINDING_IN_SCOPE"),
    }


def build(stream: ScoredStream, card: ModelCard, cfg: Config,
          deployment: str = "unnamed deployment population",
          inputs_sha256: str | None = None,
          extra_limits: list[str] | None = None,
          ordered_stream: bool = False,
          threshold: float | None = None) -> dict:
    thr = threshold if threshold is not None else card.threshold

    acc = acceptance(stream, card, cfg)
    wrk = work(stream, card, cfg, threshold=thr)
    tim = timing(stream, card, cfg, threshold=thr)
    drf = drift(stream, card, cfg,
                baseline_auroc=acc.get("measured_auroc"),
                baseline_ppv=wrk.get("row_level_ppv"),
                threshold=thr, ordered=ordered_stream)
    sub = subgroups(stream, cfg)

    unverifiable = [c.as_dict() for c in card.claims if c.kind == "unverifiable"]

    sections = {
        "acceptance": acc, "work": wrk, "timing": tim,
        "drift": drf, "subgroups": sub,
        "unverifiable_claims": {
            "question": "what does the card assert that cannot be tested?",
            "claims": unverifiable,
            "note": "recorded, never scored. A claim with no number attached is "
                    "not evidence, and its presence on a card is itself something "
                    "the committee should weigh",
        },
        "limits": {
            "same_weight_as_findings": True,
            "items": DEFAULT_LIMITS + list(extra_limits or []),
        },
    }

    record = {
        "record": cfg.record_id,
        "schema": SCHEMA,
        "subject": card.as_dict(),
        "deployment_population": deployment,
        "stream": {
            "rows": stream.n_rows,
            "entities": stream.n_entities,
            "has_time": stream.has_time,
            "has_onset": stream.has_onset,
            "prevalence_rows": round(float(stream.y.mean()), 6),
        },
        "governance_config": cfg.as_dict(),
        "headline": headline(sections),
        "sections": sections,
    }
    if inputs_sha256:
        record["inputs_sha256"] = inputs_sha256
    record = stable(record)
    record["record_sha256"] = sha256(canonical(record))
    return record
