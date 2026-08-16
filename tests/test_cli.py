"""The CLI is how this runs unattended, so the exit codes are part of the contract.

    0  no finding within scope
    1  at least one section FAILS
    2  nothing failed but something is INDETERMINATE
    3  the inputs are not verifiable
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from goodhart_monitor.cli import (EXIT_BADINPUT, EXIT_FAIL, EXIT_INDET, EXIT_OK,
                                  main)
from .conftest import make_stream

CARD = {
    "name": "TEST-1", "version": "1.0.0", "shipped_threshold": 0.5,
    "claims": [
        {"id": "M-1", "kind": "auroc", "text": "AUROC 0.80", "value": 0.80},
        {"id": "M-3", "kind": "ppv", "text": "20% of alerts are true", "value": 0.20},
    ],
}

CONFIG = """
[record]
id = "CLI-TEST"

[acceptance]
auroc_tolerance = 0.03

[drift]
windows = 4
auroc_drop = 0.05

[bootstrap]
n = 60
seed = 20260815
"""


@pytest.fixture
def workspace(tmp_path):
    """A stream, a card and a config on disk, ready for the real entry point."""
    make_stream(n_entities=120, hours=12, seed=11).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "card.json").write_text(json.dumps(CARD))
    (tmp_path / "governance.toml").write_text(CONFIG)
    return tmp_path


def run(ws, *extra):
    return main(["verify", "--stream", str(ws / "s.csv"),
                 "--card", str(ws / "card.json"),
                 "--config", str(ws / "governance.toml"),
                 "--out", str(ws / "out"), *extra])


def test_overstated_card_exits_one(workspace, capsys):
    """No stream here can support an 0.97 AUROC claim, so ACCEPTANCE must FAIL."""
    (workspace / "card.json").write_text(json.dumps(dict(CARD, claims=[
        {"id": "M-1", "kind": "auroc", "text": "AUROC 0.97", "value": 0.97},
        {"id": "M-3", "kind": "ppv", "text": "90% of alerts are true", "value": 0.90},
    ])))
    assert run(workspace) == EXIT_FAIL
    out = capsys.readouterr().out
    assert "acceptance" in out and "FAILS" in out


def test_honest_card_does_not_exit_one(workspace):
    """Re-point the card at what the stream actually measures."""
    run(workspace)
    rec = json.loads((workspace / "out" / "record_CLI-TEST.json").read_text())
    measured = rec["sections"]["acceptance"]["measured_auroc"]
    ppv = rec["sections"]["work"]["row_level_ppv"]
    honest = dict(CARD, claims=[
        {"id": "M-1", "kind": "auroc", "text": "x", "value": round(measured, 3)},
        {"id": "M-3", "kind": "ppv", "text": "x", "value": round(ppv * 0.5, 3)},
    ])
    (workspace / "card.json").write_text(json.dumps(honest))
    assert run(workspace) in (EXIT_OK, EXIT_INDET)


def test_writes_both_renderings(workspace):
    run(workspace)
    out = workspace / "out"
    assert (out / "record_CLI-TEST.json").exists()
    assert (out / "record_CLI-TEST.md").exists()
    assert "# Verification record" in (out / "record_CLI-TEST.md").read_text()


def test_rerun_is_byte_identical(workspace):
    run(workspace)
    first = (workspace / "out" / "record_CLI-TEST.json").read_bytes()
    run(workspace)
    assert (workspace / "out" / "record_CLI-TEST.json").read_bytes() == first


def test_record_id_flag_overrides_config(workspace):
    run(workspace, "--record-id", "GHM-9999")
    assert (workspace / "out" / "record_GHM-9999.json").exists()


def test_extra_limits_reach_the_record(workspace):
    run(workspace, "--limit", "single site, one calendar quarter")
    rec = json.loads((workspace / "out" / "record_CLI-TEST.json").read_text())
    assert "single site, one calendar quarter" in rec["sections"]["limits"]["items"]


def test_inputs_are_hashed_into_the_record(workspace):
    run(workspace)
    rec = json.loads((workspace / "out" / "record_CLI-TEST.json").read_text())
    assert len(rec["inputs_sha256"]) == 64


def test_editing_the_stream_changes_the_input_hash(workspace):
    run(workspace)
    a = json.loads((workspace / "out" / "record_CLI-TEST.json").read_text())
    df = pd.read_csv(workspace / "s.csv")
    df.loc[0, "score"] = min(1.0, df.loc[0, "score"] + 0.05)
    df.to_csv(workspace / "s.csv", index=False)
    run(workspace)
    b = json.loads((workspace / "out" / "record_CLI-TEST.json").read_text())
    assert a["inputs_sha256"] != b["inputs_sha256"]


# ------------------------------------------------------------------ bad input
def test_missing_stream_exits_three(workspace, capsys):
    code = main(["verify", "--stream", str(workspace / "nope.csv"),
                 "--card", str(workspace / "card.json"), "--out", str(workspace / "o")])
    assert code == EXIT_BADINPUT
    assert "cannot verify" in capsys.readouterr().err


def test_unusable_stream_exits_three_not_a_wrong_record(workspace, capsys):
    """A stream with one class must refuse, never emit a confident record."""
    df = pd.read_csv(workspace / "s.csv")
    df["label"] = 0
    df.to_csv(workspace / "s.csv", index=False)
    assert run(workspace) == EXIT_BADINPUT
    assert not (workspace / "out").exists()


def test_unsupported_format_exits_three(workspace):
    (workspace / "s.txt").write_text("nope")
    assert main(["verify", "--stream", str(workspace / "s.txt"),
                 "--card", str(workspace / "card.json"),
                 "--out", str(workspace / "o")]) == EXIT_BADINPUT


def test_card_with_no_claims_exits_three(workspace):
    (workspace / "card.json").write_text(json.dumps({"name": "x", "claims": []}))
    assert run(workspace) == EXIT_BADINPUT


# ------------------------------------------------------------------- validate
def test_validate_reports_shape(workspace, capsys):
    assert main(["validate", "--stream", str(workspace / "s.csv")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "valid stream" in out and "entities" in out
    assert "Age" in out                       # subgroup candidates surfaced


def test_validate_rejects_bad_stream(workspace):
    df = pd.read_csv(workspace / "s.csv")
    df["label"] = 7
    df.to_csv(workspace / "bad.csv", index=False)
    assert main(["validate", "--stream", str(workspace / "bad.csv")]) == EXIT_BADINPUT


# ---------------------------------------------------------------- entry point
def test_installed_console_script_runs():
    exe = Path(sys.executable).with_name("goodhart-monitor")
    if not exe.exists():
        pytest.skip("package not installed in this environment")
    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "verification records" in r.stdout
