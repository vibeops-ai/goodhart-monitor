"""psv -> parquet feature matrices, split before anything is trained.

Hospital A is the maker's world: it gets a train/dev split by patient.
Hospital B is the deployment site and is touched by NOTHING until the checker
runs. That separation is the product's whole argument, so it is enforced by
file layout rather than by promise: the maker's training script cannot even
see a path containing hospital B.
"""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_matrix  # noqa: E402

OUT = ROOT / "out"
DATA = ROOT / "data"


def split_patients(files: list[str], dev_frac: float = 0.25, seed: int = 20260815):
    rng = np.random.default_rng(seed)
    files = sorted(files)
    rng.shuffle(files)
    k = int(len(files) * dev_frac)
    return files[k:], files[:k]          # train, dev


def main() -> int:
    OUT.mkdir(exist_ok=True)
    a = DATA / "setA"
    b = DATA / "setB"
    na, nb = len(list(a.glob("p*.psv"))), len(list(b.glob("p*.psv")))
    print(f"hospital A patients: {na} · hospital B patients: {nb}")

    print("building A ...")
    dfa = build_matrix(a)
    train_ids, dev_ids = split_patients(dfa["patient"].unique().tolist())
    dfa[dfa.patient.isin(set(train_ids))].to_parquet(OUT / "A_train.parquet")
    dfa[dfa.patient.isin(set(dev_ids))].to_parquet(OUT / "A_dev.parquet")
    print(f"  A_train rows {sum(dfa.patient.isin(set(train_ids)))} · "
          f"A_dev rows {sum(dfa.patient.isin(set(dev_ids)))}")

    print("building B ...")
    dfb = build_matrix(b)
    dfb.to_parquet(OUT / "B_deploy.parquet")
    print(f"  B rows {len(dfb)}")

    # the record cites its inputs by content, not by trust
    h = hashlib.sha256()
    for f in ("A_train", "A_dev", "B_deploy"):
        h.update(hashlib.sha256((OUT / f"{f}.parquet").read_bytes()).digest())
    (OUT / "matrices.sha256").write_text(h.hexdigest() + "\n")
    print("matrices hash", h.hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
