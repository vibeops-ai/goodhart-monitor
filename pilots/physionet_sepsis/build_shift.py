"""Cut one ICU shift out of the deployment stream, for the live monitor.

The verification record is what a committee reads after the fact. This is the
thing that would actually sit in the hospital: real patients, hour by hour, with
the vendor's model scoring them and the checker watching the model.

Everything in the output is real. The vitals are the recorded vitals, the scores
are what the shipped model emitted for those exact hours, the onsets are the
challenge's adjudicated labels, and the missing values are missing because
nobody ordered that lab. Nothing is smoothed, interpolated or invented.

Two choices here are ours and are stated on the artifact itself, because a
selected cohort quietly presented as a random one is the same species of lie
this product exists to catch:

  * the cohort is enriched. A truthful random draw of a ward this size contains
    fewer than one septic patient (prevalence is 1.4% of patient-hours), and a
    shift where nothing happens demonstrates nothing. So septic stays are
    over-sampled to roughly a third, and both the cohort's prevalence and the
    deployment's real prevalence ship in the file so the page can print both.
  * the stays are staggered. Each patient's hours are their own real,
    continuous ICU hours, but patients are placed at different points in their
    stay so one shift shows a ward rather than thirty-two admissions at once.
    That is what a ward is: nobody's illness is synchronised to the clock.

The verification record's numbers are measured on all 20,000 stays and remain
the numbers to quote. This file is for showing what the work looks like.

    python pilots/physionet_sepsis/build_shift.py
    -> out/shift_GHM-0001.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"

SEED = 20260815
# A large ICU runs 40-odd beds, and ward size is not a knob to tune until the
# demo looks good: it is set once, to something a unit would recognise, and
# whatever mix of catches, misses and false alarms falls out is what is shown.
N_SEPTIC = 11
N_CLEAR = 33
SHIFT_HOURS = 12              # one nursing shift
LOOKBACK = 8                  # hours of history already on screen at handover
MIN_STAY = 20                 # a stay must cover the window it is placed in

# What a bedside nurse actually looks at, plus the two labs a sepsis workup
# turns on. Everything else in the corpus is real too and is simply not shown.
VITALS = {
    "hr": "HR", "o2": "O2Sat", "temp": "Temp", "sbp": "SBP",
    "resp": "Resp", "wbc": "WBC", "lac": "Lactate", "cr": "Creatinine",
}


def main() -> int:
    stream = pd.read_parquet(OUT / "stream_B.parquet")
    deploy = pd.read_parquet(OUT / "B_deploy.parquet")
    card = json.loads((OUT / "MODEL_CARD.json").read_text())
    thr = float(card["shipped_threshold"])

    # scores and labels live in the stream; raw observations in the matrix
    deploy = deploy.assign(entity_id=deploy["patient"].astype(str))
    joined = stream.merge(
        deploy[["entity_id", "ICULOS", *VITALS.values()]],
        left_on=["entity_id", "t"], right_on=["entity_id", "ICULOS"], how="left")

    length = joined.groupby("entity_id")["t"].max()
    ever = joined.groupby("entity_id")["label"].max()
    onset_h = joined[joined.label == 1].groupby("entity_id")["t"].min()

    span = SHIFT_HOURS + LOOKBACK
    eligible = length[length >= MIN_STAY].index

    rng = np.random.default_rng(SEED)
    # a septic stay is only worth placing if its onset can be made to fall
    # inside the shift, which is the event the unit would be living through
    septic_ok = [e for e in sorted(set(ever[ever == 1].index) & set(eligible))
                 if onset_h.get(e, 0) >= LOOKBACK + 2]
    clear_ok = sorted(set(ever[ever == 0].index) & set(eligible))
    picked_septic = list(rng.choice(septic_ok, N_SEPTIC, replace=False))
    picked_clear = list(rng.choice(clear_ok, N_CLEAR, replace=False))

    # stagger: each septic patient's onset lands at a different hour of the
    # shift, so the board is not thirty-two people deteriorating in unison
    offsets: dict[str, int] = {}
    slots = rng.permutation(np.linspace(1, SHIFT_HOURS, N_SEPTIC).round().astype(int))
    for e, slot in zip(picked_septic, slots):
        # place the stay so onset falls at shift hour `slot`
        offsets[e] = int(max(1, onset_h[e] - LOOKBACK - slot + 1))
    for e in picked_clear:
        latest = int(max(1, length[e] - span))
        offsets[e] = int(rng.integers(1, latest + 1)) if latest > 1 else 1

    pick = picked_septic + picked_clear
    rng.shuffle(pick)

    patients = []
    for pid in pick:
        off = offsets[pid]
        g = joined[(joined.entity_id == pid) & (joined.t >= off)
                   & (joined.t < off + span)].sort_values("t")
        if g.empty:
            continue
        # shift hour 0 is the handover; negative hours are the history already
        # on the screen when the shift starts
        g = g.assign(_h=g["t"].to_numpy() - off - LOOKBACK)
        onset = g.loc[g.label == 1, "_h"]
        # a stay whose onset falls outside the window is still shown; the page
        # says "no onset in this window", which is the honest thing to display
        series = []
        for _, r in g.iterrows():
            row = {"t": int(r._h), "icu": int(r.t), "s": round(float(r.score), 4)}
            for short, col in VITALS.items():
                v = r.get(col)
                if v is not None and not pd.isna(v):
                    row[short] = round(float(v), 1)
            series.append(row)
        patients.append({
            "id": f"B-{pid}",
            "age": int(g["Age"].iloc[0]),
            "sex": int(g["Gender"].iloc[0]),
            "icu_hour_at_handover": int(off + LOOKBACK),
            "onset": int(onset.min()) if len(onset) else None,
            "septic_in_stay": bool(joined.loc[joined.entity_id == pid,
                                              "label"].max() == 1),
            "series": series,
        })

    # order the board the way a unit fills: arbitrary, stable, not by outcome
    patients.sort(key=lambda p: p["id"])

    full_prevalence_rows = float(stream["label"].mean())
    doc = {
        "kind": "goodhart.monitor.shift/1",
        "record": "GHM-0001",
        "subject": {"name": card["name"], "version": card["version"]},
        "shipped_threshold": thr,
        "card": {
            "auroc": next((c["value"] for c in card["claims"]
                           if "AUROC" in c["text"]), None),
            "ppv": next((c["value"] for c in card["claims"]
                         if "alerts are true" in c["text"]), None),
        },
        "shift_hours": SHIFT_HOURS,
        "lookback_hours": LOOKBACK,
        "vitals": list(VITALS),
        "cohort": {
            "n": len(patients),
            "septic": sum(p["septic_in_stay"] for p in patients),
            "seed": SEED,
            "selection": (
                f"{N_SEPTIC} septic and {N_CLEAR} non-septic ICU stays drawn at "
                f"random (seed {SEED}) from hospital B stays of at least "
                f"{MIN_STAY} hours, each placed at a different point in its own "
                f"stay so the shift shows a ward rather than a synchronised "
                f"cohort"),
            "why_enriched": (
                "a truthful random draw of this size contains under one septic "
                "patient, and a shift where nothing happens demonstrates "
                "nothing. Rates measured on this cohort are not the "
                "deployment's rates"),
            "deployment_prevalence_rows": round(full_prevalence_rows, 6),
        },
        "patients": patients,
    }

    path = OUT / "shift_GHM-0001.json"
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")

    n_alert = sum(any(r["s"] >= thr for r in p["series"]) for p in patients)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    onsets = sorted(p["onset"] for p in patients if p["onset"] is not None)
    print(f"  {len(patients)} stays · {doc['cohort']['septic']} septic · "
          f"{SHIFT_HOURS}h shift with {LOOKBACK}h of history")
    print(f"  onsets at shift hours {onsets}")
    print(f"  {n_alert} stays alert at the shipped threshold {thr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
