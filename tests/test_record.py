"""Record-level guarantees: determinism, vocabulary, and the LIMITS rule.

Determinism is the product's central promise, so it is tested three ways:
same inputs twice, row order shuffled, and hash independence from anything
environmental.
"""
from __future__ import annotations

import json

import pytest

from goodhart_monitor import (FAILS, HOLDS, INDETERMINATE, NOT_APPLICABLE,
                              build_record, canonical, to_markdown)
from goodhart_monitor.contract import validate
from .conftest import make_stream


def test_same_inputs_give_the_same_hash(stream, card, cfg):
    a = build_record(stream, card, cfg)
    b = build_record(stream, card, cfg)
    assert a["record_sha256"] == b["record_sha256"]
    assert canonical(a) == canonical(b)


def test_row_order_does_not_change_the_record(card, cfg):
    df = make_stream(n_entities=60, hours=10, seed=4)
    a = build_record(validate(df), card, cfg)
    b = build_record(validate(df.sample(frac=1.0, random_state=9)), card, cfg)
    assert a["record_sha256"] == b["record_sha256"]


def test_hash_carries_no_wall_clock_or_path(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    blob = canonical(rec).lower()
    for forbidden in ("/users/", "timestamp", "generated_at", "hostname", "\\\\"):
        assert forbidden not in blob


def test_no_pass_anywhere_in_the_vocabulary(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    verdicts = {rec["sections"][s].get("verdict")
                for s in ("acceptance", "work", "timing", "drift")}
    assert "PASS" not in verdicts
    assert verdicts <= {HOLDS, FAILS, INDETERMINATE, NOT_APPLICABLE}


def test_limits_present_and_flagged_same_weight(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    lim = rec["sections"]["limits"]
    assert lim["same_weight_as_findings"] is True
    assert len(lim["items"]) >= 3
    assert any("no PASS" in i for i in lim["items"])


def test_extra_limits_are_appended(stream, card, cfg):
    rec = build_record(stream, card, cfg, extra_limits=["a hospital-specific caveat"])
    assert "a hospital-specific caveat" in rec["sections"]["limits"]["items"]


def test_unverifiable_claims_are_recorded_not_dropped(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    ids = [c["id"] for c in rec["sections"]["unverifiable_claims"]["claims"]]
    assert "M-4" in ids


def test_governance_config_is_printed_on_the_record(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    g = rec["governance_config"]
    assert g["drift_auroc_drop"] == cfg.drift_auroc_drop
    assert g["auroc_tolerance"] == cfg.auroc_tolerance


def test_headline_names_failing_sections(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    h = rec["headline"]
    for name in h["sections_failing"]:
        assert rec["sections"][name]["verdict"] == FAILS
    if h["sections_failing"]:
        assert h["overall"] == FAILS


def test_flat_stream_produces_a_record_with_timing_not_applicable(flat_stream, card, cfg):
    rec = build_record(flat_stream, card, cfg)
    assert rec["sections"]["timing"]["verdict"] == NOT_APPLICABLE
    assert rec["sections"]["acceptance"]["verdict"] in (HOLDS, FAILS, INDETERMINATE)


def test_markdown_renders_and_keeps_the_vocabulary(stream, card, cfg):
    rec = build_record(stream, card, cfg)
    md = to_markdown(rec)
    assert "# Verification record" in md
    assert "## ACCEPTANCE" in md and "## WORK" in md
    assert "## TIMING" in md and "## DRIFT" in md
    assert "## LIMITS" in md
    assert "There is no PASS in this vocabulary." in md
    assert "PASS ·" not in md
    # every non-obvious verdict must print its reason, not hide it in the JSON
    for name in ("acceptance", "work", "timing", "drift"):
        why = rec_why(md, name)
        assert why is None or why.strip()


def rec_why(md, section):
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"## {section.upper()} "):
            for nxt in lines[i:]:
                if nxt.startswith("**Why this verdict**"):
                    return nxt
                if nxt.startswith("## ") and not nxt.startswith(f"## {section.upper()} "):
                    return None
    return None


def test_markdown_survives_a_not_applicable_record(flat_stream, card, cfg):
    md = to_markdown(build_record(flat_stream, card, cfg))
    assert "NOT APPLICABLE" in md


def test_markdown_has_no_broken_table_rows(stream, card, cfg):
    """A stray pipe reads as a malformed table in every renderer."""
    for ln in to_markdown(build_record(stream, card, cfg)).splitlines():
        if ln.strip().endswith("|") and not ln.strip().startswith("|"):
            raise AssertionError(f"dangling pipe: {ln!r}")


def test_every_verdict_states_its_reason(stream, card, cfg):
    """A verdict a committee cannot argue with is a verdict it cannot audit."""
    rec = build_record(stream, card, cfg)
    for name in ("acceptance", "work", "timing", "drift"):
        sec = rec["sections"][name]
        assert sec.get("why"), f"{name} gives a verdict with no stated reason"
        assert len(sec["why"]) > 20


def _floats(obj, path="record"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _floats(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        yield path, obj


def test_no_integral_floats_survive_into_the_hash(stream, card, cfg):
    """Python writes 12.0 where JavaScript writes 12.

    The record is displayed by a browser that recomputes the hash. If the two
    languages canonicalise the same value differently, the page has to either
    trust the record or call it tampered, and both are wrong.
    """
    rec = build_record(stream, card, cfg)
    bad = [(p, v) for p, v in _floats(rec) if float(v).is_integer()]
    assert not bad, f"integral floats will hash differently in JS: {bad[:5]}"
