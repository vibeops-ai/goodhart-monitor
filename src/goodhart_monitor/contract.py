"""The input contract, and the reason it is the shape it is.

A verifier that needs the vendor's model cannot be deployed. Hospitals rarely
have the artifact, vendors rarely hand it over, and a checker that runs the
model itself is re-deriving outputs rather than checking the ones that actually
reached a clinician. So the contract is a **scored stream**: what the deployed
system emitted, and what later turned out to be true.

    entity_id   the unit governance counts, usually a patient or an encounter
    t           position in the stay. Hours since admission, or any monotone
                integer. Optional: without it the stream is one row per entity
                and the TIMING section reports NOT_APPLICABLE rather than
                inventing an answer
    score       the model's output, higher meaning higher risk
    label       the adjudicated outcome, 0 or 1
    onset_t     optional, when the outcome began. Absent, it is inferred as the
                first t where label == 1, which is right for time-varying
                labels and meaningless for entity-level ones, so inference only
                happens when the label actually varies within an entity

Everything else is carried through as a candidate subgroup dimension.

Validation is loud and early. A checker that silently coerces a malformed
stream produces a record that looks authoritative and is not, which is the
failure mode this company exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("entity_id", "score", "label")
RESERVED = ("entity_id", "t", "score", "label", "onset_t")


class ContractError(ValueError):
    """The stream cannot be verified. Always names the column and the fix."""


@dataclass(frozen=True)
class ScoredStream:
    """A validated deployment stream."""
    df: pd.DataFrame
    has_time: bool
    has_onset: bool
    source: str

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_entities(self) -> int:
        return int(self.df["entity_id"].nunique())

    @property
    def y(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    @property
    def p(self) -> np.ndarray:
        return self.df["score"].to_numpy()

    def subgroup_candidates(self) -> list[str]:
        return [c for c in self.df.columns if c not in RESERVED]


def _fail(msg: str) -> None:
    raise ContractError(msg)


def validate(df: pd.DataFrame, source: str = "<memory>") -> ScoredStream:
    """Check the stream is verifiable, and say precisely what is wrong if not."""
    if len(df) == 0:
        _fail(f"{source}: the stream is empty")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        _fail(f"{source}: missing required column(s) {missing}. "
              f"A scored stream needs {list(REQUIRED)}; found {list(df.columns)[:12]}")

    df = df.copy()

    # --- score
    if not pd.api.types.is_numeric_dtype(df["score"]):
        _fail(f"{source}: 'score' must be numeric, got {df['score'].dtype}")
    if df["score"].isna().any():
        n = int(df["score"].isna().sum())
        _fail(f"{source}: 'score' has {n} missing value(s). A row the model did "
              f"not score is not a row the model can be held to; drop it upstream "
              f"and say so, rather than letting the checker guess")
    if not np.isfinite(df["score"].to_numpy()).all():
        _fail(f"{source}: 'score' contains inf or -inf")

    # --- label
    lab = df["label"]
    if lab.isna().any():
        n = int(lab.isna().sum())
        _fail(f"{source}: 'label' has {n} unadjudicated row(s). Verification "
              f"needs an outcome; restrict the window to adjudicated rows")
    uniq = set(pd.unique(lab))
    if not uniq <= {0, 1, True, False, 0.0, 1.0}:
        _fail(f"{source}: 'label' must be 0/1, found values {sorted(uniq)[:6]}")
    df["label"] = lab.astype(int)

    if df["label"].nunique() < 2:
        only = int(df["label"].iloc[0])
        _fail(f"{source}: every label is {only}. Discrimination is undefined on a "
              f"single class; widen the window until both outcomes appear")

    # --- entity
    if df["entity_id"].isna().any():
        _fail(f"{source}: 'entity_id' has missing values")

    # --- time
    has_time = "t" in df.columns
    if has_time:
        if not pd.api.types.is_numeric_dtype(df["t"]):
            _fail(f"{source}: 't' must be numeric (hours since admission, or any "
                  f"monotone index), got {df['t'].dtype}")
        if df["t"].isna().any():
            _fail(f"{source}: 't' has missing values")
        dup = df.duplicated(["entity_id", "t"]).sum()
        if dup:
            _fail(f"{source}: {dup} duplicate (entity_id, t) row(s). One score per "
                  f"entity per time step, or the alert counts are wrong")
        df = df.sort_values(["entity_id", "t"], kind="stable").reset_index(drop=True)
    else:
        dup = df.duplicated(["entity_id"]).sum()
        if dup:
            _fail(f"{source}: {dup} duplicate entity_id(s) and no 't' column. "
                  f"Either add 't' for a time-varying stream, or give one row "
                  f"per entity")

    # --- onset
    has_onset = "onset_t" in df.columns
    if has_onset and not pd.api.types.is_numeric_dtype(df["onset_t"]):
        _fail(f"{source}: 'onset_t' must be numeric")
    if not has_onset and has_time:
        # only infer where the label genuinely varies inside an entity; an
        # entity-level outcome repeated on every row has no onset to find
        varies = df.groupby("entity_id")["label"].nunique().gt(1).any()
        if varies:
            onset = (df[df.label == 1].groupby("entity_id")["t"].min()
                     .rename("onset_t"))
            df = df.merge(onset, on="entity_id", how="left")
            has_onset = True

    return ScoredStream(df=df, has_time=has_time, has_onset=has_onset, source=source)


def content_sha256(stream: "ScoredStream") -> str:
    """Hash what the stream says, not the bytes it arrived in.

    Hashing the file digests the parquet container, so a writer upgrade changes
    the hash while every value is identical. The record then reports a
    difference that is not a finding, which is exactly the failure it exists to
    detect. This hashes the validated content in a fixed order with fixed
    formatting, so the same numbers give the same digest across library
    versions and file formats.
    """
    import hashlib

    df = stream.df
    cols = [c for c in df.columns if c != "onset_t"]
    h = hashlib.sha256()
    h.update(("\t".join(cols) + "\n").encode())
    for row in df[cols].itertuples(index=False, name=None):
        h.update(("\t".join(
            f"{v:.12g}" if isinstance(v, float) else str(v) for v in row
        ) + "\n").encode())
    return h.hexdigest()


def load(path: str | Path) -> ScoredStream:
    p = Path(path)
    if not p.exists():
        _fail(f"no such stream: {p}")
    if p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    else:
        _fail(f"{p}: unsupported stream format '{p.suffix}'. Use .parquet or .csv")
    return validate(df, source=p.name)
