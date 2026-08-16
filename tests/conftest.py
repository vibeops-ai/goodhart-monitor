"""Hand-built streams with known answers.

Every fixture here is small enough to verify by eye, because a test whose
expected value came out of the code it tests proves only that the code is
consistent with itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goodhart_monitor import Config
from goodhart_monitor.card import parse as parse_card
from goodhart_monitor.contract import validate


def make_stream(n_entities=200, hours=24, seed=0, signal=1.0, prevalence=0.1):
    """A synthetic time-varying stream with a controllable amount of signal.

    Positives get their label switched on at a per-entity onset hour, and their
    score is lifted from a few hours beforehand, so lead time is a real,
    known-by-construction property rather than an accident of noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        pos = rng.random() < prevalence
        onset = int(rng.integers(6, hours - 2)) if pos else None
        for t in range(1, hours + 1):
            base = rng.normal(0.3, 0.12)
            if pos and onset is not None and t >= onset - 4:
                base += signal * 0.35
            rows.append({
                "entity_id": f"e{e:04d}",
                "t": t,
                "score": float(np.clip(base, 0, 1)),
                "label": int(bool(pos and onset is not None and t >= onset)),
                "Age": 40 + (e % 50),
                "Gender": e % 2,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def stream():
    return validate(make_stream(), source="fixture")


@pytest.fixture
def flat_stream():
    """One row per entity, no time: the readmission shape."""
    rng = np.random.default_rng(1)
    n = 300
    lab = (rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({
        "entity_id": [f"e{i:04d}" for i in range(n)],
        "score": np.clip(rng.normal(0.3, 0.1, n) + lab * 0.25, 0, 1),
        "label": lab,
    })
    return validate(df, source="flat")


@pytest.fixture
def card():
    return parse_card({
        "name": "TEST-1", "version": "1.0.0", "shipped_threshold": 0.5,
        "claims": [
            {"id": "M-1", "kind": "auroc", "text": "AUROC 0.80", "value": 0.80},
            {"id": "M-2", "kind": "lead_time", "text": "predicts before onset",
             "value": 6.0},
            {"id": "M-3", "kind": "ppv", "text": "20% of alerts are true",
             "value": 0.20},
            {"id": "M-4", "kind": "unverifiable",
             "text": "generalises to new hospitals"},
        ],
    }, source="fixture-card")


@pytest.fixture
def cfg():
    return Config(bootstrap_n=60, drift_windows=4, min_cell=5).validate()
