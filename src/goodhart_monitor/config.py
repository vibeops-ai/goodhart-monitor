"""Governance thresholds, owned by the hospital and printed on the record.

Every judgement the checker makes is a threshold somebody chose. Leaving those
buried in module constants would make the record's verdicts look like physics.
They are policy, they belong to the committee, and the record carries the exact
values that produced it so a reader can disagree with the policy rather than
having to reverse-engineer it.

Defaults come from what the two buyers said out loud: Tignanelli's "5%
deterioration in the AUROC triggers review", and Singh's insistence that alert
burden be counted, not asserted.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Subgroup:
    column: str
    bins: list[float] | None = None
    labels: list[str] | None = None
    values: dict[str, str] | None = None


@dataclass(frozen=True)
class Config:
    record_id: str = "GHM-0001"
    # ACCEPTANCE: how far under the card's number still counts as holding
    auroc_tolerance: float = 0.03
    # WORK: measured PPV must reach this fraction of the card's claim
    ppv_tolerance_fraction: float = 0.8
    # TIMING: share of catches that must precede onset for the claim to hold
    min_share_before_onset: float = 0.5
    actionable_window_hours: float = 12.0
    # DRIFT: movement against the LOCAL baseline, never against the card
    drift_windows: int = 10
    drift_auroc_drop: float = 0.05
    drift_ppv_floor_fraction: float = 0.5
    # consecutive triggered windows before a trigger becomes a verdict. A single
    # window is noise plus a nudge to look; a run is a finding
    drift_review_run_for_fail: int = 2
    # bootstrap
    bootstrap_n: int = 400
    bootstrap_seed: int = 20260815
    # minimum cell size below which a metric is INDETERMINATE, not a number
    min_cell: int = 50
    subgroups: list[Subgroup] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["subgroups"] = [
            {k: v for k, v in s.items() if v is not None} for s in d["subgroups"]]
        return d

    def validate(self) -> "Config":
        if not 0 <= self.auroc_tolerance <= 0.5:
            raise ConfigError("auroc_tolerance must be within 0..0.5")
        if not 0 < self.ppv_tolerance_fraction <= 1:
            raise ConfigError("ppv_tolerance_fraction must be within (0, 1]")
        if not 0 <= self.min_share_before_onset <= 1:
            raise ConfigError("min_share_before_onset must be within 0..1")
        if self.drift_review_run_for_fail < 1:
            raise ConfigError("drift_review_run_for_fail must be at least 1")
        if self.drift_review_run_for_fail > self.drift_windows:
            raise ConfigError(
                "drift_review_run_for_fail cannot exceed drift_windows: no run "
                "that long could ever be observed")
        if self.drift_windows < 2:
            raise ConfigError("drift_windows must be at least 2 to show movement")
        if self.bootstrap_n < 50:
            raise ConfigError("bootstrap_n below 50 gives an interval nobody should quote")
        for s in self.subgroups:
            if s.bins is not None:
                if s.labels is None or len(s.labels) != len(s.bins) - 1:
                    raise ConfigError(
                        f"subgroup '{s.column}': {len(s.bins)} bin edges need "
                        f"{len(s.bins) - 1} labels")
            elif s.values is None:
                raise ConfigError(
                    f"subgroup '{s.column}': give either bins+labels or values")
        return self


DEFAULT = Config().validate()


def load(path: str | Path | None) -> Config:
    if path is None:
        return DEFAULT
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"no such config: {p}")
    raw = tomllib.loads(p.read_text())

    flat: dict = {}
    flat.update(raw.get("record", {}))
    flat.update(raw.get("acceptance", {}))
    flat.update(raw.get("work", {}))
    flat.update(raw.get("timing", {}))
    for k, v in raw.get("drift", {}).items():
        flat[f"drift_{k}" if not k.startswith("drift_") else k] = v
    for k, v in raw.get("bootstrap", {}).items():
        flat[f"bootstrap_{k}" if not k.startswith("bootstrap_") else k] = v
    flat.update(raw.get("limits", {}))

    if "id" in flat:
        flat["record_id"] = flat.pop("id")
    if "windows" in flat:
        flat["drift_windows"] = flat.pop("windows")
    if "n" in flat:
        flat["bootstrap_n"] = flat.pop("n")
    if "review_run_for_fail" in flat:
        flat["drift_review_run_for_fail"] = flat.pop("review_run_for_fail")
    if "seed" in flat:
        flat["bootstrap_seed"] = flat.pop("seed")

    subs = [Subgroup(**s) for s in raw.get("subgroups", [])]
    known = {f.name for f in Config.__dataclass_fields__.values()}
    unknown = set(flat) - known
    if unknown:
        raise ConfigError(f"{p.name}: unknown setting(s) {sorted(unknown)}")
    return Config(**{k: v for k, v in flat.items()}, subgroups=subs).validate()
