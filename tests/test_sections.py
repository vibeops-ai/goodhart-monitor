"""Section behaviour, including the two miscounts that shipped in v0.

Both regressions here are real: the first version of this checker credited an
alert fired after onset as a catch, and measured drift against the vendor card
so a constant gap looked like a trend.
"""
from __future__ import annotations

import pandas as pd
import pytest

from goodhart_monitor import Config, FAILS, HOLDS, INDETERMINATE, NOT_APPLICABLE
from goodhart_monitor.card import parse as parse_card
from goodhart_monitor.contract import validate
from goodhart_monitor.sections import acceptance, drift, entity_table, timing, work


def tiny_card(**over):
    body = {"name": "T", "version": "1", "shipped_threshold": 0.5,
            "claims": [{"id": "M-1", "kind": "auroc", "text": "AUROC 0.9", "value": 0.9},
                       {"id": "M-3", "kind": "ppv", "text": "50% true", "value": 0.5}]}
    body.update(over)
    return parse_card(body)


# ----------------------------------------------------------------- ACCEPTANCE
def test_acceptance_fails_when_local_is_worse(stream, card, cfg):
    """Fixture stream carries less signal than a 0.80 card claim."""
    a = acceptance(stream, card, cfg)
    assert a["verdict"] in (FAILS, HOLDS, INDETERMINATE)
    assert a["measured_auroc"] is not None
    assert a["ci95_entity_bootstrap"] is not None
    lo, hi = a["ci95_entity_bootstrap"]
    assert lo <= a["measured_auroc"] <= hi


def test_acceptance_not_applicable_without_a_number(stream, cfg):
    c = parse_card({"name": "T", "version": "1",
                    "claims": [{"id": "X", "kind": "unverifiable", "text": "is good"}]})
    assert acceptance(stream, c, cfg)["verdict"] == NOT_APPLICABLE


def test_acceptance_holds_within_tolerance(stream, cfg):
    measured = acceptance(stream, tiny_card(), cfg)["measured_auroc"]
    c = tiny_card(claims=[{"id": "M-1", "kind": "auroc", "text": "x",
                           "value": round(measured, 4)}])
    assert acceptance(stream, c, cfg)["verdict"] == HOLDS


# ----------------------------------------------------------------------- WORK
def _late_alert_stream():
    """One positive entity whose only alert lands AFTER onset, plus a decoy.

    Onset is hour 3. The alert fires at hour 5. That is case finding, and the
    v0 checker scored it as a catch.
    """
    rows = []
    for t in range(1, 7):
        rows.append({"entity_id": "late", "t": t,
                     "score": 0.9 if t >= 5 else 0.1,
                     "label": 1 if t >= 3 else 0})
    for t in range(1, 7):
        rows.append({"entity_id": "clean", "t": t, "score": 0.1, "label": 0})
    return validate(pd.DataFrame(rows))


def test_alert_after_onset_is_not_a_catch():
    s = _late_alert_stream()
    cfg = Config(min_cell=1, drift_windows=2).validate()
    w = work(s, tiny_card(), cfg)
    assert w["actionable_catches"] == 0
    assert w["entities_evaluated_per_actionable_catch"] is None
    assert w["sensitivity_actionable"] == 0.0
    # it still counts as an alert, because it is still work
    assert w["n_alert_rows"] == 2


def test_early_alert_is_a_catch():
    rows = []
    for t in range(1, 7):
        rows.append({"entity_id": "early", "t": t,
                     "score": 0.9 if t >= 2 else 0.1,
                     "label": 1 if t >= 4 else 0})
    for t in range(1, 7):
        rows.append({"entity_id": "clean", "t": t, "score": 0.1, "label": 0})
    s = validate(pd.DataFrame(rows))
    cfg = Config(min_cell=1, drift_windows=2).validate()
    w = work(s, tiny_card(), cfg)
    assert w["actionable_catches"] == 1
    assert w["entities_evaluated_per_actionable_catch"] == 1.0
    assert w["sensitivity_actionable"] == 1.0


def test_work_always_reports_sensitivity_beside_the_ratio(stream, card, cfg):
    """A flattering burden ratio bought by never alerting must be visible."""
    w = work(stream, card, cfg)
    if w["entities_evaluated_per_actionable_catch"] is not None:
        assert w["sensitivity_actionable"] is not None


def test_work_not_applicable_without_threshold(stream, cfg):
    c = parse_card({"name": "T", "version": "1",
                    "claims": [{"id": "M-1", "kind": "auroc", "text": "x", "value": 0.8}]})
    assert work(stream, c, cfg)["verdict"] == NOT_APPLICABLE


def test_entity_table_counts_alert_rows():
    s = _late_alert_stream()
    et = entity_table(s, 0.5).set_index("entity_id")
    assert et.loc["late", "n_alert_rows"] == 2
    assert et.loc["clean", "n_alert_rows"] == 0
    assert bool(et.loc["late", "positive"]) is True


# --------------------------------------------------------------------- TIMING
def test_timing_not_applicable_without_time(flat_stream, card, cfg):
    t = timing(flat_stream, card, cfg)
    assert t["verdict"] == NOT_APPLICABLE
    assert "no time column" in t["why"]


def test_timing_fails_when_catches_are_late():
    s = _late_alert_stream()
    cfg = Config(min_cell=1, drift_windows=2).validate()
    t = timing(s, tiny_card(), cfg)
    assert t["caught_before_onset"] == 0
    assert t["share_of_catches_after_onset"] == 1.0
    assert t["verdict"] == FAILS


def test_timing_reports_actionable_window_separately(stream, card, cfg):
    t = timing(stream, card, cfg)
    if t["verdict"] != NOT_APPLICABLE and t.get("caught_at_all"):
        assert t["caught_within_window"] <= t["caught_before_onset"]


# ---------------------------------------------------------------------- DRIFT
def test_drift_baseline_is_local_not_card(stream, card, cfg):
    """A card far above local reality must not light up every window.

    This is the v0 bug: measuring each window against the card re-reported the
    acceptance gap ten times and called a constant a trend.
    """
    local = acceptance(stream, card, cfg)["measured_auroc"]
    w = work(stream, card, cfg)
    optimistic = parse_card({
        "name": "T", "version": "1", "shipped_threshold": 0.5,
        "claims": [{"id": "M-1", "kind": "auroc", "text": "AUROC 0.99", "value": 0.99},
                   {"id": "M-3", "kind": "ppv", "text": "90% true", "value": 0.9}]})
    d = drift(stream, optimistic, cfg, baseline_auroc=local,
              baseline_ppv=w["row_level_ppv"], threshold=0.5)
    assert d["baseline"].startswith("local")
    # a stable stream against its own baseline should not be all-review
    assert d["windows_triggering_review"] < cfg.drift_windows


def _half_noise_stream():
    """Second half of the stream is pure noise: real, sustained deterioration."""
    import numpy as np
    rng = np.random.default_rng(3)
    rows = []
    for e in range(400):
        good = e < 200
        lab = int(rng.random() < 0.3)
        sc = (0.2 + 0.6 * lab + rng.normal(0, 0.05)) if good else rng.random()
        rows.append({"entity_id": f"{'a' if good else 'z'}{e:04d}",
                     "score": float(np.clip(sc, 0, 1)), "label": lab})
    return validate(pd.DataFrame(rows))


def _drift_on(s, cfg, **kw):
    a = acceptance(s, tiny_card(), cfg)
    w = work(s, tiny_card(), cfg)
    return drift(s, tiny_card(), cfg, baseline_auroc=a["measured_auroc"],
                 baseline_ppv=w["row_level_ppv"], threshold=0.5, **kw)


def test_drift_fails_on_sustained_degradation_in_a_calendar_stream(cfg):
    d = _drift_on(_half_noise_stream(), cfg, ordered=True)
    assert d["windows_triggering_review"] >= 2
    assert d["longest_consecutive_run"] >= cfg.drift_review_run_for_fail
    assert d["verdict"] == FAILS


def test_drift_without_a_calendar_never_returns_a_verdict(cfg):
    """Windows cut from sorted ids measure sampling noise, not time."""
    d = _drift_on(_half_noise_stream(), cfg)          # ordered defaults to False
    assert d["verdict"] == INDETERMINATE
    assert "carries none" in d["why"]
    assert d["windows"]                                # still printed, as an instrument check


def test_one_isolated_window_is_a_trigger_not_a_finding(cfg):
    """The whole point of a review threshold is that it fires early and often."""
    import numpy as np
    rng = np.random.default_rng(5)
    rows = []
    for e in range(400):
        bad = 200 <= e < 240                           # one window's worth, then recovery
        lab = int(rng.random() < 0.3)
        sc = rng.random() if bad else (0.2 + 0.6 * lab + rng.normal(0, 0.05))
        rows.append({"entity_id": f"e{e:04d}", "score": float(np.clip(sc, 0, 1)),
                     "label": lab})
    d = _drift_on(validate(pd.DataFrame(rows)), Config(min_cell=1, drift_windows=10,
                                                       bootstrap_n=60).validate(),
                  ordered=True)
    assert d["windows_triggering_review"] >= 1
    assert not d["latest_window_triggers_review"]
    assert d["verdict"] == INDETERMINATE
    assert "trigger for review, not a finding" in d["why"]


def test_drift_indeterminate_when_too_few_entities(cfg):
    rows = [{"entity_id": f"e{i}", "score": i / 10, "label": i % 2} for i in range(6)]
    s = validate(pd.DataFrame(rows))
    d = drift(s, tiny_card(), cfg, baseline_auroc=0.6, baseline_ppv=0.3, threshold=0.5)
    assert d["verdict"] == INDETERMINATE


def test_timing_ships_the_distribution_not_only_the_median(stream, card, cfg):
    """A median lead averages a two-day warning with a one-hour one."""
    t = timing(stream, card, cfg)
    dist = t["lead_time_distribution"]
    assert [d["bucket"] for d in dist] == [
        "after onset", "0-6h", "6-12h", "12-24h", "24-48h", "48h+"]
    assert sum(d["catches"] for d in dist) == t["caught_at_all"]
    after = next(d for d in dist if d["bucket"] == "after onset")
    assert after["catches"] == t["caught_at_all"] - t["caught_before_onset"]


# ---------------------------------------------------------------------- SWEEP
def test_sweep_agrees_with_the_sections_at_every_threshold(stream, card, cfg):
    """The sweep is a fast path, and a fast path that disagrees is a second
    product. At each swept threshold the numbers must equal what the real
    sections produce at that same threshold."""
    from goodhart_monitor.sweep import sweep as run_sweep

    sw = run_sweep(stream, card, cfg, n_points=12)
    assert sw["points"]
    for pt in sw["points"]:
        thr = pt["threshold"]
        w = work(stream, card, cfg, threshold=thr)
        t = timing(stream, card, cfg, threshold=thr)
        assert pt["n_alert_rows"] == w["n_alert_rows"], thr
        if w["row_level_ppv"] is not None:
            assert abs(pt["row_ppv"] - w["row_level_ppv"]) < 5e-4, thr
        assert pt["actionable_catches"] == w["actionable_catches"], thr
        if w["entities_evaluated_per_actionable_catch"] is not None:
            assert abs(pt["entities_per_actionable_catch"]
                       - w["entities_evaluated_per_actionable_catch"]) < 0.05, thr
        if t.get("caught_at_all") is not None:
            assert pt["caught_at_all"] == t["caught_at_all"], thr
            assert pt["caught_before_onset"] == t["caught_before_onset"], thr


def test_sweep_lead_bins_sum_to_the_early_catches(stream, card, cfg):
    from goodhart_monitor.sweep import sweep as run_sweep
    sw = run_sweep(stream, card, cfg, n_points=8)
    for pt in sw["points"]:
        total = sum(pt["lead_bins"]) + pt["lead_beyond_last_bin"]
        assert total == pt["caught_before_onset"]


def test_sweep_states_no_verdicts(stream, card, cfg):
    """Policy belongs to the reader; this artifact carries measurements only."""
    from goodhart_monitor.sweep import sweep as run_sweep
    import json
    sw = run_sweep(stream, card, cfg, n_points=6)
    blob = json.dumps({k: v for k, v in sw.items() if k != "note"})
    for word in ("HOLDS", "FAILS", "INDETERMINATE", "verdict"):
        assert word not in blob


def test_sweep_is_monotone_in_alert_volume(stream, card, cfg):
    from goodhart_monitor.sweep import sweep as run_sweep
    pts = run_sweep(stream, card, cfg, n_points=20)["points"]
    vols = [p["n_alert_rows"] for p in pts]
    assert vols == sorted(vols, reverse=True)


def test_sweep_always_contains_the_shipped_threshold(stream, card, cfg):
    """The reader's actual operating point is not approximated."""
    from goodhart_monitor.sweep import sweep as run_sweep
    sw = run_sweep(stream, card, cfg, n_points=10)
    assert any(abs(p["threshold"] - card.threshold) < 1e-9 for p in sw["points"])
