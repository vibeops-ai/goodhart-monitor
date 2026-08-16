"""PhysioNet/CinC 2019 sepsis data -> one feature row per patient-hour.

The corpus: 40,336 real ICU stays from two separate hospital systems, one
pipe-separated file per patient, one row per hour. 40 columns of vitals, labs
and demographics, plus SepsisLabel, which the challenge organisers already
shifted so that a positive label appears roughly six hours BEFORE clinical
sepsis onset. That shift is what makes "does it warn early or does it just
notice afterwards" a measurable question rather than a marketing sentence.

Feature policy, chosen to mirror what a real vendor ships rather than to win
the challenge:

  * last observation carried forward within a stay, because that is what a
    deployed model sees at the bedside;
  * a measured-recently indicator per lab, because missingness in ICU data is
    informational (nobody orders a lactate on a patient who looks fine);
  * no future information of any kind: features at hour t use rows <= t only.

Nothing here imputes across patients and nothing normalises using statistics
from the evaluation hospital. The maker is trained on hospital A alone; if a
statistic leaks from hospital B into training, the acceptance record that the
checker produces would be measuring contamination, not transfer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "BUN",
        "Creatinine", "Glucose", "Lactate", "Magnesium", "Potassium",
        "Hct", "Hgb", "WBC", "Platelets"]
STATIC = ["Age", "Gender", "HospAdmTime"]
CLOCK = ["ICULOS"]
LABEL = "SepsisLabel"


def load_patient(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df["patient"] = path.stem
    return df


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    """One stay -> one row per hour, causal features only."""
    out = pd.DataFrame(index=df.index)
    for c in VITALS + LABS:
        ff = df[c].ffill()
        out[c] = ff
        out[f"{c}_missing"] = df[c].isna().astype(np.int8)
        # hours since this value was last actually measured
        measured = df[c].notna()
        grp = measured.cumsum()
        out[f"{c}_age"] = (~measured).groupby(grp).cumsum().astype(np.float32)
    for c in VITALS:  # short-horizon trend on the things measured hourly
        out[f"{c}_d1"] = out[c].diff()
    for c in STATIC + CLOCK:
        out[c] = df[c]
    out[LABEL] = df[LABEL].astype(np.int8)
    out["patient"] = df["patient"].iloc[0]
    return out


def build_matrix(folder: Path, limit: int | None = None) -> pd.DataFrame:
    files = sorted(folder.glob("p*.psv"))
    if limit:
        files = files[:limit]
    frames = [featurise(load_patient(f)) for f in files]
    return pd.concat(frames, ignore_index=True)


FEATURES: list[str] | None = None


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (LABEL, "patient")]
