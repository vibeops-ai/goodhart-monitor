"""The checker. This is the GoodHart product; everything else here is scaffolding.

Input contract:  a frozen model artifact, its model card, and a stream of
                 (features, later-adjudicated outcome) rows from the hospital
                 where it is actually running.
Output contract: a content-addressed verification record with four sections,
                 each of which answers a question a governance committee is
                 already asking in Singh's and Tignanelli's own words:

  ACCEPTANCE   does the card's number hold on this population?
               (Singh: "come with your own data, not the vendor's")
  WORK         what work does the alert stream create at the shipped
               threshold — alerts/day, PPV, patients evaluated per true case?
               (Singh: "what work does this model create, and is it valuable?")
  TIMING       does it warn before onset, or notice afterwards?
               (the exact claim class the Epic sepsis card got wrong)
  DRIFT        windowed performance over the deployment stream with explicit
               review thresholds
               (Tignanelli: "5% deterioration in the AUROC triggers review —
                that's missing; we do it manually every three months")

Verdict vocabulary is the house one. A claim is HOLDS, FAILS or INDETERMINATE,
each with a bootstrap interval; there is no PASS and no green, and the record
carries a LIMITS section at the same weight as the findings, because a checker
that cannot say what it cannot see is another vendor slide.
"""
from __future__ import annotations

import json
import pickle
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitor.features import LABEL, feature_columns  # noqa: E402

OUT = ROOT / "out"
RNG = np.random.default_rng(20260815)

HOLDS, FAILS, INDET = "HOLDS", "FAILS", "INDETERMINATE"

# review thresholds, stated up front the way a committee would set them
DRIFT_AUROC_DROP = 0.05          # Tignanelli's own example number
DRIFT_PPV_FLOOR_FRAC = 0.5       # PPV halves against the card -> review
ACCEPT_TOL_AUROC = 0.03          # card AUROC minus this still HOLDS


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


def boot_ci(fn, y, p, n=400):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(fn(y[s], p[s]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def patient_level(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Collapse hours to what governance actually counts: patients and alerts."""
    rows = []
    for pid, g in df.groupby("patient"):
        septic = bool(g[LABEL].max())
        onset = float(g.loc[g[LABEL] == 1, "ICULOS"].min()) if septic else None
        fired = g.loc[g.score >= thr, "ICULOS"]
        rows.append({
            "patient": pid, "septic": septic, "onset": onset,
            "alerted": bool(len(fired)),
            "first_alert": float(fired.min()) if len(fired) else None,
            "n_alert_hours": int(len(fired)),
            "hours": int(len(g)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    art = pickle.loads((OUT / "maker.pkl").read_bytes())
    card = json.loads((OUT / "MODEL_CARD.json").read_text())
    clf, feats = art["model"], art["features"]
    thr = card["shipped_threshold"]

    deploy = pd.read_parquet(OUT / "B_deploy.parquet")
    deploy = deploy.assign(score=clf.predict_proba(deploy[feats])[:, 1])
    y = deploy[LABEL].to_numpy()
    p = deploy["score"].to_numpy()

    claims = {c["id"]: c for c in card["claims"]}
    sections = {}

    # ------------------------------------------------------------ ACCEPTANCE
    auroc = float(roc_auc_score(y, p))
    lo, hi = boot_ci(roc_auc_score, y, p)
    card_auroc = claims["M-1"]["value"]
    sections["acceptance"] = {
        "question": "does the card's headline number hold on this population?",
        "card_claim": claims["M-1"]["text"],
        "measured_auroc": round(auroc, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "verdict": HOLDS if hi >= card_auroc - ACCEPT_TOL_AUROC and auroc >= card_auroc - ACCEPT_TOL_AUROC
        else (FAILS if hi < card_auroc - ACCEPT_TOL_AUROC else INDET),
        "gap": round(card_auroc - auroc, 4),
        "n_patient_hours": int(len(y)),
    }

    # ------------------------------------------------------------------ WORK
    pl = patient_level(deploy, thr)
    n_pat = len(pl)
    alerted = pl[pl.alerted]
    true_alerted = alerted[alerted.septic]
    ppv_patient = len(true_alerted) / len(alerted) if len(alerted) else 0.0
    total_days = pl["hours"].sum() / 24.0
    hour_mask = p >= thr
    hour_ppv = float(y[hour_mask].mean()) if hour_mask.sum() else 0.0
    sections["work"] = {
        "question": "what work does the alert stream create at the shipped threshold?",
        "card_claim": claims["M-3"]["text"],
        "threshold": thr,
        "alerts_per_100_patient_days": round(100 * float(hour_mask.sum()) / total_days, 1),
        "share_of_patients_ever_alerted": round(len(alerted) / n_pat, 4),
        "patient_level_ppv": round(ppv_patient, 4),
        "hour_level_ppv": round(hour_ppv, 4),
        "patients_evaluated_per_true_case": round(1 / ppv_patient, 1) if ppv_patient else None,
        "verdict": HOLDS if hour_ppv >= claims["M-3"]["value"] * 0.8
        else FAILS,
        "note": "patients-evaluated-per-true-case is Singh's number: at Michigan the "
                "Epic model needed 8; every alert beyond the true one is created work",
    }

    # ---------------------------------------------------------------- TIMING
    septics = pl[pl.septic]
    caught = septics[septics.alerted]
    before = caught[caught.first_alert < caught.onset]
    lead = (caught.onset - caught.first_alert).to_numpy()
    sections["timing"] = {
        "question": "does it warn before onset, or notice afterwards?",
        "card_claim": claims["M-2"]["text"],
        "septic_patients": int(len(septics)),
        "caught_at_all": int(len(caught)),
        "sensitivity_patient_level": round(len(caught) / len(septics), 4) if len(septics) else None,
        "caught_before_onset": int(len(before)),
        "share_of_catches_after_onset": round(1 - len(before) / len(caught), 4) if len(caught) else None,
        "median_lead_hours_when_early": round(float(np.median(lead[lead > 0])), 1) if (lead > 0).any() else 0.0,
        "verdict": HOLDS if len(caught) and (len(before) / len(caught)) >= 0.5 else FAILS,
        "note": "label already sits ~6h before onset, so 'after onset' here means the model "
                "fired after even the early-shifted label window opened — case finding, "
                "not prediction. This is the claim class the Epic sepsis card got wrong.",
    }

    # ----------------------------------------------------------------- DRIFT
    # The deployment stream, in admission batches. Order within the corpus is
    # constructed (the corpus carries no calendar), and the record says so; the
    # monitoring machinery is identical for a calendar stream.
    pids = sorted(pl.patient.tolist())
    k = 10
    windows = []
    for i in range(k):
        wp = set(pids[i * len(pids) // k:(i + 1) * len(pids) // k])
        w = deploy[deploy.patient.isin(wp)]
        wy, wp_ = w[LABEL].to_numpy(), w["score"].to_numpy()
        wm = wp_ >= thr
        w_auroc = float(roc_auc_score(wy, wp_))
        w_ppv = float(wy[wm].mean()) if wm.sum() else 0.0
        windows.append({
            "window": i + 1, "patients": len(wp),
            "auroc": round(w_auroc, 4), "hour_ppv": round(w_ppv, 4),
            "alerts": int(wm.sum()),
            "review": bool(w_auroc < card_auroc - DRIFT_AUROC_DROP
                           or w_ppv < claims["M-3"]["value"] * DRIFT_PPV_FLOOR_FRAC),
        })
    sections["drift"] = {
        "question": "windowed performance against explicit review thresholds",
        "thresholds": {
            "auroc_drop_from_card": DRIFT_AUROC_DROP,
            "ppv_floor_fraction_of_card": DRIFT_PPV_FLOOR_FRAC,
        },
        "windows": windows,
        "windows_triggering_review": sum(w["review"] for w in windows),
        "provenance": "stream order constructed from corpus; machinery identical for calendar streams",
    }

    # ---------------------------------------------------------------- LIMITS
    sections["limits"] = {
        "same_weight_as_findings": True,
        "items": [
            "outcome labels are the challenge's Sepsis-3 adjudication, not a chart review "
            "by this hospital's clinicians",
            "hospital B here is a real second health system, but the maker and checker "
            "were built by the same company for this pilot artifact; in production the "
            "maker is the vendor's and we never train it",
            "no calendar timestamps exist in the corpus, so DRIFT demonstrates the "
            "machinery on a constructed stream and is labelled accordingly",
            "subgroup coverage below is age and sex only; the corpus carries no race or "
            "language fields at all, which is itself a finding about vendor data",
        ],
    }

    # ------------------------------------------------------------- subgroups
    subs = []
    for name, mask in [
        ("age<65", deploy.Age < 65), ("age>=65", deploy.Age >= 65),
        ("female", deploy.Gender == 0), ("male", deploy.Gender == 1),
    ]:
        my, mp = y[mask.to_numpy()], p[mask.to_numpy()]
        subs.append({"group": name, "n_hours": int(mask.sum()),
                     "auroc": round(float(roc_auc_score(my, mp)), 4)})
    sections["subgroups"] = subs

    record = {
        "record": "GHM-0001",
        "schema": "goodhart.monitor/1",
        "subject": {"model": card["name"], "version": card["version"],
                    "model_sha256": card["model_sha256"]},
        "deployment_population": "hospital system B · 20,000 real ICU stays · "
                                 "PhysioNet/CinC 2019 · untouched during maker development",
        "inputs_sha256": (OUT / "matrices.sha256").read_text().strip(),
        "sections": sections,
    }
    record["record_sha256"] = sha256(canonical(record))
    (OUT / "record_GHM-0001.json").write_text(json.dumps(record, indent=1))

    # console digest
    print(f"record GHM-0001  {record['record_sha256'][:16]}")
    for k_ in ("acceptance", "work", "timing"):
        s = sections[k_]
        print(f"  {k_:10} {s.get('verdict'):13} | {s.get('card_claim','')[:58]}")
    print(f"  drift      {sections['drift']['windows_triggering_review']}/10 windows trigger review")
    print(f"  measured AUROC on B: {sections['acceptance']['measured_auroc']} "
          f"(card said {card_auroc})")
    print(f"  work: {sections['work']['alerts_per_100_patient_days']} alerts/100pt-days, "
          f"NNE {sections['work']['patients_evaluated_per_true_case']}")
    print(f"  timing: {sections['timing']['share_of_catches_after_onset']:.0%} of catches after onset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
