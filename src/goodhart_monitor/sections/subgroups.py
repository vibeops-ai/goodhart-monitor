"""SUBGROUPS — where the number is worse, and which dimensions are absent.

Two findings live here, and the second is the one people forget. Reporting
performance by age and sex is useful. Reporting that the stream carries no race,
ethnicity or language column at all is often more useful, because it means
nobody can answer the equity question about this deployment — including the
vendor who sold it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import stats
from ..config import Config, Subgroup
from ..contract import ScoredStream

# dimensions a committee will ask about; absence is reported as a finding
EXPECTED_DIMENSIONS = {
    "race": ("race", "ethnicity"),
    "language": ("language", "preferred_language", "lang"),
    "insurance": ("insurance", "payer", "coverage"),
}


def _assign(df: pd.DataFrame, s: Subgroup) -> pd.Series | None:
    if s.column not in df.columns:
        return None
    col = df[s.column]
    if s.bins is not None:
        return pd.cut(col, bins=s.bins, labels=s.labels, right=False,
                      include_lowest=True).astype(object)
    return col.astype(str).map(s.values)


def subgroups(stream: ScoredStream, cfg: Config) -> dict:
    df = stream.df
    rows = []
    missing_cols = []

    for s in cfg.subgroups:
        assigned = _assign(df, s)
        if assigned is None:
            missing_cols.append(s.column)
            continue
        for name, idx in assigned.dropna().groupby(assigned.dropna()).groups.items():
            sub = df.loc[idx]
            y, p = sub["label"].to_numpy(), sub["score"].to_numpy()
            a = stats.auroc(y, p)
            rows.append({
                "dimension": s.column,
                "group": str(name),
                "n_rows": int(len(sub)),
                "n_entities": int(sub["entity_id"].nunique()),
                "prevalence": round(float(y.mean()), 4),
                "auroc": None if a is None else round(a, 4),
                "underpowered": bool(len(sub) < cfg.min_cell or a is None),
            })

    present = {c.lower() for c in df.columns}
    absent = [k for k, alts in EXPECTED_DIMENSIONS.items()
              if not any(a in present for a in alts)]

    spread = None
    valid = [r["auroc"] for r in rows if r["auroc"] is not None and not r["underpowered"]]
    if len(valid) > 1:
        spread = round(float(max(valid) - min(valid)), 4)

    return {
        "question": "where is the number worse, and which dimensions cannot be asked?",
        "groups": rows,
        "auroc_spread_across_groups": spread,
        "configured_columns_absent_from_stream": missing_cols,
        "dimensions_the_stream_cannot_answer": absent,
        "note": "a dimension the stream does not carry is a finding, not a blank. "
                "If race, language or payer are absent, nobody can answer the "
                "equity question about this deployment, the vendor included",
    }
