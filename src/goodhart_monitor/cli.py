"""Command line entry point.

    goodhart-monitor verify --stream S --card C [--config F] [--out DIR]

Exit codes are meaningful, because this belongs in a scheduled job as much as
in a meeting:

    0  no finding within scope
    1  at least one section FAILS
    2  no failure, but something is INDETERMINATE and needs a human
    3  the inputs are not verifiable (contract or card error)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import config as cfgmod
from . import contract, record, render, stats
from .sweep import sweep as run_sweep
from . import intake as intakemod
from . import runner as runnermod
from .card import CardError, load as load_card
from .contract import ContractError

EXIT_OK, EXIT_FAIL, EXIT_INDET, EXIT_BADINPUT = 0, 1, 2, 3


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def cmd_verify(a: argparse.Namespace) -> int:
    try:
        stream = contract.load(a.stream)
        card = load_card(a.card)
        cfg = cfgmod.load(a.config)
    except (ContractError, CardError, cfgmod.ConfigError) as e:
        print(f"cannot verify: {e}", file=sys.stderr)
        return EXIT_BADINPUT

    if a.record_id:
        cfg = cfgmod.Config(**{**cfg.as_dict(), "record_id": a.record_id,
                               "subgroups": cfg.subgroups}).validate()

    rec = record.build(
        stream, card, cfg,
        deployment=a.deployment,
        inputs_sha256=_hash_files([Path(a.stream), Path(a.card)]),
        extra_limits=a.limit or None,
        ordered_stream=a.ordered,
        threshold=a.threshold)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"record_{cfg.record_id}.json"
    mpath = out / f"record_{cfg.record_id}.md"
    jpath.write_text(json.dumps(rec, indent=1) + "\n")
    mpath.write_text(render.to_markdown(rec))

    h = rec["headline"]
    print(f"record {cfg.record_id}  {rec['record_sha256'][:16]}")
    for name in ("acceptance", "work", "timing", "drift"):
        sec = rec["sections"][name]
        print(f"  {name:11} {sec.get('verdict', '?'):15} "
              f"{(sec.get('card_claim') or sec.get('question') or '')[:56]}")
    print(f"  overall     {h['overall']}")
    print(f"  wrote {jpath.name}, {mpath.name}")

    if h["sections_failing"]:
        return EXIT_FAIL
    if h["tally"].get(stats.INDETERMINATE):
        return EXIT_INDET
    return EXIT_OK


def cmd_validate(a: argparse.Namespace) -> int:
    try:
        s = contract.load(a.stream)
    except ContractError as e:
        print(f"invalid: {e}", file=sys.stderr)
        return EXIT_BADINPUT
    print(f"valid stream: {s.n_rows:,} rows · {s.n_entities:,} entities · "
          f"time={s.has_time} onset={s.has_onset}")
    print(f"  prevalence {s.y.mean():.4f}")
    print(f"  subgroup candidates: {', '.join(s.subgroup_candidates()[:12]) or 'none'}")
    return EXIT_OK


def cmd_sweep(a: argparse.Namespace) -> int:
    """Every operating point, so a committee can choose one with its eyes open."""
    try:
        stream = contract.load(a.stream)
        card = load_card(a.card)
        cfg = cfgmod.load(a.config)
    except (ContractError, CardError, cfgmod.ConfigError) as e:
        print(f"cannot sweep: {e}", file=sys.stderr)
        return EXIT_BADINPUT

    sw = run_sweep(stream, card, cfg, n_points=a.points)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sweep_{cfg.record_id}.json"
    path.write_text(json.dumps(sw, separators=(",", ":")) + "\n")

    print(f"sweep {cfg.record_id}: {len(sw['points'])} operating points")
    print(f"  AUROC {sw['auroc']} does not move with the threshold")
    print(f"  wrote {path.name} ({path.stat().st_size // 1024} KB)")
    return EXIT_OK


def cmd_intake(a: argparse.Namespace) -> int:
    """Answer 'can you run this on our export' before any verdict exists."""
    try:
        m = intakemod.load(a.manifest)
    except intakemod.IntakeError as e:
        print(f"manifest rejected: {e}", file=sys.stderr)
        return EXIT_BADINPUT
    r = intakemod.assess(m)
    print(intakemod.render(r))
    return EXIT_OK if r.runnable else EXIT_BADINPUT


def cmd_run(a: argparse.Namespace) -> int:
    """Run the verifier over a hospital export named by a manifest."""
    try:
        m = intakemod.load(a.manifest)
    except intakemod.IntakeError as e:
        print(f"manifest rejected: {e}", file=sys.stderr)
        return EXIT_BADINPUT
    r = intakemod.assess(m)
    if not r.runnable:
        print(intakemod.render(r), file=sys.stderr)
        return EXIT_BADINPUT

    out = Path(a.out)
    res = runnermod.run(m, out, record_id=a.record_id)
    rep = res["report"]
    print(f"verified {res['events']:,} events -> {res['api']}")
    for k, label in (("coverage", "coverage"), ("confirmed_validity", "validity"),
                     ("landing", "landing")):
        met = rep[k]
        val = "n/a" if met["value"] is None else f"{met['value']:.4f}"
        print(f"  {label:<9} {val:>8}   {met['numerator']}/{met['denominator']}")
    print(f"  evc       {'withheld' if rep['evc'] is None else rep['evc']:>8}")
    for f in rep["findings"]:
        print(f"  - {f}")
    return EXIT_OK


def cmd_selftest(a: argparse.Namespace) -> int:
    """Prove the install works end to end, with synthetic data and no access."""
    dest = Path(a.dir)
    mp = intakemod.synthesise(dest, with_outcomes=not a.no_outcomes)
    print(f"wrote synthetic export and manifest to {dest}/")
    m = intakemod.load(mp)
    r = intakemod.assess(m)
    print()
    print(intakemod.render(r))
    if not r.runnable:
        return EXIT_BADINPUT
    print()
    res = runnermod.run(m, dest / "api", record_id="SELFTEST")
    rep = res["report"]
    print(f"verified {res['events']:,} synthetic events")
    print(f"  coverage {rep['coverage']['value']}  "
          f"validity {rep['confirmed_validity']['value']}  "
          f"landing {rep['landing']['value']}  "
          f"evc {'withheld' if rep['evc'] is None else rep['evc']}")
    print(f"  api written to {dest / 'api'}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="goodhart-monitor",
        description="Independent verification records for deployed clinical models.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="produce a verification record")
    v.add_argument("--stream", required=True,
                   help="scored deployment stream (.parquet or .csv)")
    v.add_argument("--card", required=True, help="the vendor's model card (.json)")
    v.add_argument("--config", default=None, help="governance thresholds (.toml)")
    v.add_argument("--out", default="out", help="output directory")
    v.add_argument("--deployment", default="unnamed deployment population")
    v.add_argument("--record-id", default=None)
    v.add_argument("--threshold", type=float, default=None,
                   help="override the card's shipped threshold")
    v.add_argument("--ordered", action="store_true",
                   help="the stream is already in calendar order; drift will say so")
    v.add_argument("--limit", action="append",
                   help="an extra LIMITS line; repeatable")
    v.set_defaults(fn=cmd_verify)

    s_ = sub.add_parser("sweep", help="measure every operating point, state no verdicts")
    s_.add_argument("--stream", required=True)
    s_.add_argument("--card", required=True)
    s_.add_argument("--config", default=None)
    s_.add_argument("--out", default="out")
    s_.add_argument("--points", type=int, default=220)
    s_.set_defaults(fn=cmd_sweep)

    i = sub.add_parser("intake", help="check a hospital export against the manifest")
    i.add_argument("--manifest", required=True)
    i.set_defaults(fn=cmd_intake)

    rn = sub.add_parser("run", help="verify a hospital export named by a manifest")
    rn.add_argument("--manifest", required=True)
    rn.add_argument("--out", default="out/api")
    rn.add_argument("--record-id", default="GHM-LOCAL")
    rn.set_defaults(fn=cmd_run)

    st = sub.add_parser("selftest", help="run the whole loop on synthetic data")
    st.add_argument("--dir", default="selftest")
    st.add_argument("--no-outcomes", action="store_true",
                    help="omit the outcome export, to see validity degrade")
    st.set_defaults(fn=cmd_selftest)

    c = sub.add_parser("validate", help="check a stream against the input contract")
    c.add_argument("--stream", required=True)
    c.set_defaults(fn=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
