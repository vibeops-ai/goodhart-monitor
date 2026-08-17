"""Intake decides whether a hospital's export can be verified at all.

The failure messages matter as much as the success path: an analyst who gets
"missing column" with the column list can fix it without calling us.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodhart_monitor import intake, runner


@pytest.fixture
def export(tmp_path):
    return intake.synthesise(tmp_path / "x", n_subjects=12, hours=14, seed=3)


def test_synthetic_export_is_ready(export):
    r = intake.assess(intake.load(export))
    assert r.runnable
    assert not r.blocking
    assert all(c["available"] for c in r.checks if c["required"])


def test_report_names_every_role(export):
    r = intake.assess(intake.load(export))
    tables = {x["table"] for x in r.roles}
    assert tables == set(intake.ROLES)
    actions = next(x for x in r.roles if x["table"] == "actions")
    assert actions["availability"] == "not_collected"


def test_missing_required_table_blocks(tmp_path, export):
    m = export.read_text().replace('[population_context]', '[population_context]\navailability = "requires_approval"')
    export.write_text(m)
    r = intake.assess(intake.load(export))
    assert not r.runnable
    assert any("population_context is requires_approval" in b for b in r.blocking)


def test_unmapped_vital_codes_block_completeness(tmp_path, export):
    body = export.read_text()
    body = body.split("[codes]")[0] + '[codes]\nHEART_RATE = "HR"\n'
    export.write_text(body)
    r = intake.assess(intake.load(export))
    chk = next(c for c in r.checks if c["check_id"] == "chk.input-completeness")
    assert not chk["available"]
    assert "O2Sat" in chk["why"] and "SBP" in chk["why"]
    assert not r.runnable          # the check is required by the policy


def test_bad_column_name_names_the_columns_present(tmp_path, export):
    body = export.read_text().replace('score        = "sepsis_risk"',
                                      'score        = "risk_of_sepsis"')
    export.write_text(body)
    with pytest.raises(intake.IntakeError) as e:
        intake.load(export)
    msg = str(e.value)
    assert "risk_of_sepsis" in msg and "Present:" in msg and "sepsis_risk" in msg


def test_unknown_availability_is_rejected(export):
    export.write_text(export.read_text().replace(
        'availability = "not_collected"', 'availability = "maybe"'))
    with pytest.raises(intake.IntakeError) as e:
        intake.load(export)
    assert "not a contract state" in str(e.value)


def test_bad_mode_is_rejected(export):
    export.write_text(export.read_text().replace(
        'mode            = "monitoring"', 'mode            = "shadow"'))
    with pytest.raises(intake.IntakeError) as e:
        intake.load(export)
    assert "not a contract mode" in str(e.value)


def test_missing_manifest():
    with pytest.raises(intake.IntakeError):
        intake.load("/nonexistent/manifest.toml")


# ------------------------------------------------------------------- runner
def test_run_produces_a_validated_api(tmp_path, export):
    m = intake.load(export)
    res = runner.run(m, tmp_path / "api", record_id="T")
    assert res["events"] > 0
    assert res["validated"] is True          # schema present in this checkout
    idx = json.loads((tmp_path / "api" / "index.json").read_text())
    assert len(idx["rows"]) == res["events"]
    for row in idx["rows"][:20]:
        assert (tmp_path / "api" / "events" / f"{row['event_id']}.json").exists()


def test_landing_is_zero_when_the_review_queue_is_unstaffed(tmp_path, export):
    """Flags route to review, so Landing has a denominator. Nothing closes it,
    so the rate is 0 and EVC is 0. An unstaffed queue is a measurement, not a
    missing measurement."""
    res = runner.run(intake.load(export), tmp_path / "api")
    assert res["landing"]["denominator"] > 0
    assert res["landing"]["numerator"] == 0
    assert res["landing"]["value"] == 0.0
    assert res["evc"] == 0.0


def test_validity_is_none_without_an_outcome_export(tmp_path):
    mp = intake.synthesise(tmp_path / "y", n_subjects=10, hours=12,
                           with_outcomes=False)
    res = runner.run(intake.load(mp), tmp_path / "api")
    assert res["validity"]["denominator"] == 0
    assert res["validity"]["value"] is None
    assert res["evc"] is None
    assert any("Coverage counts outputs the verifier received" in f
               for f in res["report"]["findings"])


def test_two_thresholds_in_one_export_are_refused(tmp_path, export):
    import pandas as pd
    d = export.parent
    df = pd.read_csv(d / "ai_outputs.csv")
    export.write_text(export.read_text().replace(
        "threshold    = 0.32", 'threshold    = "thr"'))
    df["thr"] = [0.3 if i % 2 else 0.4 for i in range(len(df))]
    df.to_csv(d / "ai_outputs.csv", index=False)
    with pytest.raises(SystemExit) as e:
        runner.run(intake.load(export), tmp_path / "api")
    assert "more than one threshold" in str(e.value)


# ------------------------------------------------ product-path parity
def test_runner_uses_the_same_metric_code_as_the_pilot(tmp_path, export):
    """The library and the demonstration must compute metrics identically. A
    demonstration with better methodology than the library is two products."""
    res = runner.run(intake.load(export), tmp_path / "api")
    md = res["report"]["metadata"]
    for key in ("validity_basis", "validity_per_event", "validity_decomposition",
                "validity_counting_inconclusive_as_missed", "concentration",
                "events_excluded_outside_period", "coverage_denominator"):
        assert key in md, f"{key} missing from the runner's report"
    assert res["report"]["confirmed_validity"]["denominator"] > 0
    assert md["validity_basis"].startswith("per patient")


def test_landing_requires_the_closure_state_the_policy_demands(tmp_path, export):
    """An auto-close job must not be able to move Landing. The policy asks for
    'reviewed' on a flag; a 'delivered' event alone must not count."""
    from goodhart_monitor import periodic
    m = intake.load(export)
    res = runner.run(m, tmp_path / "api")
    import json
    idx = json.loads((tmp_path / "api" / "index.json").read_text())
    flagged = [r for r in idx["rows"] if r["verdict"] == "flag"]
    assert flagged, "expected some flags in the synthetic export"
    # every flag was delivered, none reviewed
    assert all(r["disposition"] == "delivered" for r in flagged)
    assert res["landing"]["denominator"] == len(flagged)
    assert res["landing"]["numerator"] == 0
