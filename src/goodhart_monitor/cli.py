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

    c = sub.add_parser("validate", help="check a stream against the input contract")
    c.add_argument("--stream", required=True)
    c.set_defaults(fn=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
