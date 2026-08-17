"""Turn hospital B's deployment matrix into a scored stream.

This is the only step in the pilot that needs the vendor's model, and it is
deliberately on the pilot side of the wall, not in the product. A real hospital
never runs this: the scores are already in the EHR, written there by whatever
the vendor deployed, and the checker reads them out. The score is already
logged per patient-hour, and verifying that output stream rather than the model
behind it is what makes the check independent of the vendor.

Here we simulate that log by scoring hospital B once with the shipped maker and
writing exactly the six columns the contract asks for. Nothing else crosses:
not the features, not the model, not the training data. If this file were
deleted and replaced by a CSV export from Epic, the rest of the pilot would run
unchanged, which is the property that makes the product sellable.

    python pilots/physionet_sepsis/to_stream.py
    -> out/stream_B.parquet   entity_id, t, score, label, Age, Gender
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import LABEL  # noqa: E402

OUT = ROOT / "out"


def main() -> int:
    deploy = pd.read_parquet(OUT / "B_deploy.parquet")
    bundle = pickle.loads((OUT / "maker.pkl").read_bytes())
    model, feats = bundle["model"], bundle["features"]

    score = model.predict_proba(deploy[feats])[:, 1]

    stream = pd.DataFrame({
        # a hospital would use its own MRN or CSN here; the column name is all
        # the contract cares about
        "entity_id": deploy["patient"].astype(str),
        "t": deploy["ICULOS"].astype(int),
        "score": score.astype(float),
        "label": deploy[LABEL].astype(int),
        # subgroups: only what a hospital could actually attach to an alert log
        "Age": deploy["Age"].astype(float),
        "Gender": deploy["Gender"].astype(int),
    }).sort_values(["entity_id", "t"], kind="mergesort").reset_index(drop=True)

    path = OUT / "stream_B.parquet"
    stream.to_parquet(path, index=False)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path.relative_to(ROOT)}")
    print(f"  {len(stream):,} patient-hours · {stream.entity_id.nunique():,} patients")
    print(f"  positive-label share {stream.label.mean():.4f}")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
