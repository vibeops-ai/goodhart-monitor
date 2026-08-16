"""Metrics, intervals, verdict boundaries, and card parsing."""
from __future__ import annotations

import numpy as np
import pytest

from goodhart_monitor import stats
from goodhart_monitor.card import CardError, parse
from goodhart_monitor.config import Config, ConfigError


# ------------------------------------------------------------------- metrics
def test_auroc_is_none_on_a_single_class():
    assert stats.auroc(np.zeros(10), np.random.rand(10)) is None


def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    assert stats.auroc(y, p) == 1.0
    assert stats.auroc(y, -p) == 0.0


def test_ppv_is_none_when_nothing_alerts():
    """Zero would read as 'measured, and terrible'. None reads as 'no evidence'."""
    y = np.array([0, 1, 0, 1])
    assert stats.ppv(y, np.zeros(4, dtype=bool)) is None


def test_ppv_counts_only_alerts():
    y = np.array([0, 1, 1, 0]); m = np.array([True, True, False, False])
    assert stats.ppv(y, m) == 0.5


def test_cluster_bootstrap_is_wider_than_a_row_bootstrap():
    """Rows inside an entity are correlated; ignoring that reports false precision.

    Each entity contributes 30 near-identical rows, so the stream has 1800 rows
    but only 60 independent observations. A row bootstrap believes it has 1800.
    Distributions overlap on purpose: a perfectly separable stream pins AUROC at
    1.0 and both intervals collapse to a point, which measures nothing.
    """
    rng = np.random.default_rng(0)
    ent, y, p = [], [], []
    for e in range(60):
        lab = int(rng.random() < 0.4)
        centre = rng.normal(0.4 + 0.15 * lab, 0.18)   # entity-level effect, overlapping
        for _ in range(30):
            ent.append(e); y.append(lab)
            p.append(centre + rng.normal(0, 0.01))    # rows barely differ
    ent, y, p = np.array(ent), np.array(y), np.array(p)
    assert 0.55 < stats.auroc(y, p) < 0.95            # genuinely uncertain, not saturated
    clus = stats.entity_bootstrap_ci(y, p, ent, n=300, seed=1)
    rows = stats.entity_bootstrap_ci(y, p, np.arange(len(y)), n=300, seed=1)
    assert (clus[1] - clus[0]) > 2 * (rows[1] - rows[0])


def test_bootstrap_is_seed_deterministic():
    rng = np.random.default_rng(2)
    ent = np.repeat(np.arange(40), 5)
    y = (rng.random(200) < 0.4).astype(int)
    p = rng.random(200)
    a = stats.entity_bootstrap_ci(y, p, ent, n=80, seed=7)
    b = stats.entity_bootstrap_ci(y, p, ent, n=80, seed=7)
    assert a == b


def test_bootstrap_refuses_tiny_entity_counts():
    ent = np.arange(5); y = np.array([0, 1, 0, 1, 0]); p = np.random.rand(5)
    assert stats.entity_bootstrap_ci(y, p, ent, n=50, seed=0) is None


# ------------------------------------------------------------------ verdicts
@pytest.mark.parametrize("measured,ci,expected", [
    (0.80, (0.79, 0.83), stats.HOLDS),            # interval clears the floor
    (0.60, (0.58, 0.62), stats.FAILS),            # interval entirely below
    (0.76, (0.72, 0.82), stats.INDETERMINATE),    # straddles the floor
    (None, None, stats.INDETERMINATE),            # nothing measurable
])
def test_verdict_boundaries(measured, ci, expected):
    assert stats.verdict_at_least(measured, 0.80, 0.03, ci) == expected


def test_verdict_without_interval_uses_the_point():
    assert stats.verdict_at_least(0.78, 0.80, 0.03, None) == stats.HOLDS
    assert stats.verdict_at_least(0.70, 0.80, 0.03, None) == stats.FAILS


# ---------------------------------------------------------------------- card
def test_card_infers_kind_from_legacy_text():
    c = parse({"name": "x", "claims": [
        {"id": "a", "text": "AUROC 0.81 for sepsis", "value": 0.81},
        {"id": "b", "text": "predicts sepsis before onset", "value": 12},
        {"id": "c", "text": "~20% of alerts are true", "value": 0.2},
        {"id": "d", "text": "generalises everywhere"},
    ]})
    assert [x.kind for x in c.claims] == ["auroc", "lead_time", "ppv", "unverifiable"]


def test_card_rejects_impossible_rates():
    with pytest.raises(CardError):
        parse({"name": "x", "claims": [{"id": "a", "kind": "auroc",
                                        "text": "", "value": 1.4}]})


def test_card_rejects_duplicate_ids():
    with pytest.raises(CardError):
        parse({"name": "x", "claims": [{"id": "a", "text": "t", "value": 0.5},
                                       {"id": "a", "text": "t", "value": 0.5}]})


def test_card_rejects_no_claims():
    with pytest.raises(CardError) as e:
        parse({"name": "x", "claims": []})
    assert "cannot be verified" in str(e.value)


def test_card_accepts_threshold_under_either_key():
    assert parse({"name": "x", "threshold": 0.4,
                  "claims": [{"id": "a", "text": "t", "value": 0.5}]}).threshold == 0.4
    assert parse({"name": "x", "shipped_threshold": 0.6,
                  "claims": [{"id": "a", "text": "t", "value": 0.5}]}).threshold == 0.6


# -------------------------------------------------------------------- config
@pytest.mark.parametrize("kw", [
    {"auroc_tolerance": 0.9}, {"ppv_tolerance_fraction": 0},
    {"min_share_before_onset": 2}, {"drift_windows": 1}, {"bootstrap_n": 10},
])
def test_config_rejects_nonsense(kw):
    with pytest.raises(ConfigError):
        Config(**kw).validate()


def test_config_round_trips_to_dict():
    c = Config().validate()
    assert c.as_dict()["drift_auroc_drop"] == 0.05
