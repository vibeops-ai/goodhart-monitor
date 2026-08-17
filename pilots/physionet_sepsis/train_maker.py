"""The maker: a sepsis early-warning model, built the way a vendor builds one.

This is deliberately a competent, ordinary model — gradient-boosted trees on
causal bedside features, the same family as most commercial early-warning
products — not a straw man and not a challenge winner. It trains on hospital A
only, picks its operating threshold on its own dev split, and then writes the
kind of model card a vendor writes: its own numbers, measured on its own data,
phrased the way the market phrases them.

Every number in the card is real. That is the point of the exercise: the card
is not a lie, and the checker will still find that three of its five claims do
not survive contact with a hospital the model has never seen. That gap — true
numbers, wrong implications — is the thing GoodHart sells the measurement of,
and it is the gap external validation found in the Epic sepsis model card.
"""
from __future__ import annotations

import json
import pickle
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import LABEL, feature_columns  # noqa: E402

OUT = ROOT / "out"


def main() -> int:
    train = pd.read_parquet(OUT / "A_train.parquet")
    dev = pd.read_parquet(OUT / "A_dev.parquet")
    feats = feature_columns(train)

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0)
    clf.fit(train[feats], train[LABEL])

    p_dev = clf.predict_proba(dev[feats])[:, 1]
    auroc = roc_auc_score(dev[LABEL], p_dev)
    auprc = average_precision_score(dev[LABEL], p_dev)

    # vendor-style threshold choice: the point where dev-set precision is ~20%,
    # i.e. one true septic patient-hour per five alerts, a common shipping point
    qs = np.quantile(p_dev, np.linspace(0.90, 0.9995, 400))
    thr, ppv = None, None
    for t in qs:
        mask = p_dev >= t
        if mask.sum() < 50:
            break
        prec = dev[LABEL][mask].mean()
        if prec >= 0.20:
            thr, ppv = float(t), float(prec)
            break
    if thr is None:
        thr = float(np.quantile(p_dev, 0.999))
        ppv = float(dev[LABEL][p_dev >= thr].mean())

    # patient-level dev lead time, the number the market loves
    dev = dev.assign(score=p_dev)
    lead = []
    for _, g in dev.groupby("patient"):
        if g[LABEL].max() == 0:
            continue
        onset = g.loc[g[LABEL] == 1, "ICULOS"].min()
        fired = g.loc[g.score >= thr, "ICULOS"]
        if len(fired):
            lead.append(onset - fired.min())
    lead = np.array(lead)
    med_lead = float(np.median(lead)) if len(lead) else 0.0

    (OUT / "maker.pkl").write_bytes(pickle.dumps({"model": clf, "features": feats}))
    model_sha = hashlib.sha256((OUT / "maker.pkl").read_bytes()).hexdigest()

    card = {
        "name": "MAKER-1 sepsis early warning",
        "version": "1.0.0",
        "model_sha256": model_sha,
        "intended_use": "hourly sepsis risk on adult ICU patients",
        "training_data": "hospital system A (development split held out)",
        "claims": [
            {"id": "M-1", "text": f"AUROC {auroc:.2f} for sepsis prediction",
             "value": round(float(auroc), 4), "population": "vendor dev set"},
            {"id": "M-2", "text": "predicts sepsis before clinical onset",
             "value": round(med_lead, 1),
             "detail": f"median lead time {med_lead:.0f} h on vendor dev set"},
            {"id": "M-3", "text": f"~{ppv:.0%} of alerts are true at the shipped threshold",
             "value": round(float(ppv), 4), "population": "vendor dev set"},
            {"id": "M-4", "text": "performance generalises to new hospital systems",
             "value": None, "detail": "no supporting measurement provided"},
        ],
        "shipped_threshold": thr,
        "dev_auprc": round(float(auprc), 4),
    }
    (OUT / "MODEL_CARD.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
