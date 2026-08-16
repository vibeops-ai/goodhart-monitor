"""The vendor's model card, parsed as a set of falsifiable claims.

A card is only checkable to the extent it says something that could be wrong.
So each claim declares its `kind`, and the checker knows how to test each kind:

    auroc        a discrimination number, tested by ACCEPTANCE
    ppv          alert precision at the shipped threshold, tested by WORK
    lead_time    warns before onset, tested by TIMING
    unverifiable anything asserted without a number, e.g. "generalises to new
                 hospitals". Recorded, never scored, and reported as such —
                 because an unfalsifiable claim on a card is itself a finding

Unknown kinds are preserved verbatim and reported unverifiable rather than
dropped. Silently ignoring a claim we cannot test would let a vendor put the
important sentence in a field we happen not to read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KINDS = {"auroc", "ppv", "lead_time", "unverifiable"}


class CardError(ValueError):
    pass


@dataclass(frozen=True)
class Claim:
    id: str
    kind: str
    text: str
    value: float | None = None
    detail: str | None = None
    population: str | None = None

    def as_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "text": self.text}
        if self.value is not None:
            d["value"] = self.value
        if self.detail:
            d["detail"] = self.detail
        if self.population:
            d["population"] = self.population
        return d


@dataclass(frozen=True)
class ModelCard:
    name: str
    version: str
    threshold: float | None
    claims: list[Claim] = field(default_factory=list)
    model_sha256: str | None = None
    intended_use: str | None = None
    training_data: str | None = None

    def of_kind(self, kind: str) -> Claim | None:
        for c in self.claims:
            if c.kind == kind:
                return c
        return None

    def as_dict(self) -> dict:
        d = {"name": self.name, "version": self.version,
             "claims": [c.as_dict() for c in self.claims]}
        for k in ("threshold", "model_sha256", "intended_use", "training_data"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def _infer_kind(c: dict) -> str:
    """Back-compat for cards written before `kind` existed."""
    if c.get("kind") in KINDS:
        return c["kind"]
    t = (c.get("text") or "").lower()
    if c.get("value") is None:
        return "unverifiable"
    if "auroc" in t or "auc" in t or "c-statistic" in t:
        return "auroc"
    if "alert" in t or "precision" in t or "ppv" in t or "true" in t:
        return "ppv"
    if "before" in t or "lead" in t or "predict" in t or "early" in t:
        return "lead_time"
    return "unverifiable"


def parse(obj: dict, source: str = "<memory>") -> ModelCard:
    for req in ("name", "claims"):
        if req not in obj:
            raise CardError(f"{source}: model card missing '{req}'")
    if not isinstance(obj["claims"], list) or not obj["claims"]:
        raise CardError(f"{source}: 'claims' must be a non-empty list. A card that "
                        f"claims nothing cannot be verified, and that is a finding "
                        f"to report to the committee, not a file to check")

    claims = []
    seen: set[str] = set()
    for i, c in enumerate(obj["claims"]):
        cid = c.get("id") or f"C-{i + 1}"
        if cid in seen:
            raise CardError(f"{source}: duplicate claim id '{cid}'")
        seen.add(cid)
        kind = _infer_kind(c)
        val = c.get("value")
        if kind in ("auroc", "ppv") and val is not None and not (0.0 <= float(val) <= 1.0):
            raise CardError(f"{source}: claim {cid} is a {kind} of {val}; expected 0..1")
        claims.append(Claim(
            id=cid, kind=kind, text=c.get("text", ""),
            value=None if val is None else float(val),
            detail=c.get("detail"), population=c.get("population")))

    thr = obj.get("shipped_threshold", obj.get("threshold"))
    return ModelCard(
        name=obj["name"], version=str(obj.get("version", "unversioned")),
        threshold=None if thr is None else float(thr), claims=claims,
        model_sha256=obj.get("model_sha256"), intended_use=obj.get("intended_use"),
        training_data=obj.get("training_data"))


def load(path: str | Path) -> ModelCard:
    p = Path(path)
    if not p.exists():
        raise CardError(f"no such model card: {p}")
    return parse(json.loads(p.read_text()), source=p.name)
