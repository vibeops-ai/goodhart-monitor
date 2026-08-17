"""Run the verifier over a hospital export and write the read API.

Same check packs, same decision table and same contract objects the pilot uses.
The only difference is where the rows come from: a manifest instead of the
PhysioNet parquet.

Degradation is explicit. No outcome export means every verdict stays
unresolved and Confirmed Validity has no denominator. No action export means
Landing has none either. In both cases EVC is withheld and the report says
which factor was missing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import contracts as C
from .contracts import Row, base, ref
from . import periodic
from .intake import Manifest, ROLES

RESOLUTION_HOURS = 48
SCHEMA_PATH = (Path(__file__).resolve().parents[2]
               / "contracts" / "v2" / "goodhart-verifier-contracts.schema.json")


def _validator():
    """Validate every emitted object, when the schema and jsonschema are here.

    A packaged install without the schema still runs; the CLI reports that
    validation was skipped so nobody assumes it happened.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return None
    if not SCHEMA_PATH.exists():
        return None
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def _col(t, name):
    return t.df[t.mapped[name]]


def _iso(x) -> str:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def build_rows(m: Manifest) -> tuple[pd.DataFrame, dict, dict, float]:
    """Join the export into one row per scored output, in ICU-hour space."""
    ai = m.tables["ai_outputs"]
    obs = m.tables["observations"]
    ctx = m.tables["population_context"]

    s = pd.DataFrame({
        "subject": _col(ai, "subject_ref").astype(str),
        "at": pd.to_datetime(_col(ai, "occurred_at"), utc=True, format="mixed"),
        "score": pd.to_numeric(_col(ai, "score"), errors="coerce"),
    })
    if "event_id" in ai.mapped:
        s["event_id"] = _col(ai, "event_id").astype(str)
    thr_series = (pd.to_numeric(_col(ai, "threshold"), errors="coerce")
                  if "threshold" in ai.mapped else None)
    if thr_series is None:
        raise SystemExit("manifest maps no threshold on [ai_outputs]; the "
                         "verifier cannot replay an alert decision without it")
    threshold = float(thr_series.iloc[0])
    if thr_series.nunique(dropna=True) > 1:
        raise SystemExit("the export carries more than one threshold. Split the "
                         "run by threshold, or the replay check is meaningless")

    if s["score"].isna().any():
        raise SystemExit(f"{int(s['score'].isna().sum())} scores are not numeric")

    # hour index per subject, so the runtime speaks the same units everywhere
    s = s.sort_values(["subject", "at"], kind="mergesort").reset_index(drop=True)
    first = s.groupby("subject")["at"].transform("min")
    s["hour"] = ((s["at"] - first).dt.total_seconds() // 3600).astype(int)

    o = pd.DataFrame({
        "subject": _col(obs, "subject_ref").astype(str),
        "obs_at": pd.to_datetime(_col(obs, "observed_at"), utc=True, format="mixed"),
        "code": _col(obs, "code").astype(str).map(lambda c: m.codes.get(c, c)),
        "value": pd.to_numeric(_col(obs, "value"), errors="coerce"),
    }).dropna(subset=["value"])

    c = pd.DataFrame({"subject": _col(ctx, "subject_ref").astype(str)})
    c["age"] = (pd.to_numeric(_col(ctx, "age"), errors="coerce")
                if "age" in ctx.mapped else np.nan)
    c["sex"] = (pd.to_numeric(_col(ctx, "sex"), errors="coerce").fillna(-1)
                if "sex" in ctx.mapped else -1)
    ctx_map = c.drop_duplicates("subject").set_index("subject")

    onset: dict[str, int] = {}
    if "outcomes" in m.tables:
        out = m.tables["outcomes"]
        od = pd.DataFrame({
            "subject": _col(out, "subject_ref").astype(str),
            "at": pd.to_datetime(_col(out, "outcome_at"), utc=True, format="mixed"),
        })
        base_at = s.groupby("subject")["at"].min()
        for sub, g in od.groupby("subject"):
            if sub in base_at.index:
                h = int((g["at"].min() - base_at[sub]).total_seconds() // 3600)
                onset[sub] = h
    return s, {"obs": o, "ctx": ctx_map, "onset": onset}, {}, threshold


def run(m: Manifest, out_dir: Path, record_id: str = "GHM-LOCAL",
        entity_ppv: float | None = None) -> dict:
    s, aux, _, threshold = build_rows(m)
    obs, ctx_map, onset = aux["obs"], aux["ctx"], aux["onset"]
    d = m.deployment

    C.ORG = d["organization_id"]
    C.SITE = d["site_id"]
    C.SYSTEM_ID = d["system_id"]
    C.DEPLOYMENT_ID = d["deployment_id"]

    card = {
        "name": d.get("model_name", d["system_id"]),
        "version": d.get("model_version", "unknown"),
        "model_sha256": d.get("model_sha256", "0" * 64),
        "shipped_threshold": threshold,
        "claims": d.get("claims", [
            {"id": "M-1", "text": f"AUROC {d.get('claimed_auroc', 0.8)}",
             "value": float(d.get("claimed_auroc", 0.8))},
            {"id": "M-3", "text": f"{d.get('claimed_ppv', 0.2):.0%} of alerts are true",
             "value": float(d.get("claimed_ppv", 0.2))},
            {"id": "M-2", "text": "predicts before onset", "value": 0.0},
        ]),
    }
    claims = {c["id"]: c for c in card["claims"]}
    for need in ("M-1", "M-2", "M-3"):
        claims.setdefault(need, {"id": need, "text": "not stated", "value": None})
    card["claims"] = [claims[k] for k in ("M-1", "M-2", "M-3")]

    ccs = C.claim_contracts(card)
    ab = C.authority_bundle()
    fmp = C.failure_mode_profile()
    policy = C.verification_policy(threshold, None)
    policy["mode"] = d["mode"]

    # first alert per subject, needed by truth resolution
    alerting = s[s.score >= threshold]
    first_alert = alerting.groupby("subject")["hour"].min().to_dict()

    obs_by_subject = {k: v.sort_values("obs_at") for k, v in obs.groupby("subject")}
    ppv = entity_ppv
    if ppv is None and onset:
        alerted = set(first_alert)
        ppv = (sum(1 for a in alerted if a in onset) / len(alerted)) if alerted else 0.0
    ppv = float(ppv or 0.0)

    validator = _validator()
    invalid: list[str] = []

    def keep(obj, what):
        if validator is not None:
            errs = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
            if errs:
                e = errs[0]
                invalid.append(f"{what}: {'/'.join(str(x) for x in e.path)}: "
                               f"{e.message}")
        return obj

    events, checks, verdicts, disps, truths = [], [], [], [], []
    rng = np.random.default_rng(20260815)

    for rec in s.itertuples():
        sub, hour = rec.subject, int(rec.hour)
        eid = getattr(rec, "event_id", None) or f"ev.{sub}.{hour:04d}"
        eid = str(eid).replace(" ", "-")
        trace = f"tr.{eid}"

        og = obs_by_subject.get(sub)
        vitals: dict[str, tuple[float | None, float | None]] = {}
        # everything any check pack can use, not only the bedside four
        for code in sorted(set(C.REQUIRED_VITALS) | set(C.OBTAINABLE_INPUTS)):
            v = a = None
            if og is not None:
                cut = og[(og["code"] == code) & (og["obs_at"] <= rec.at)]
                if len(cut):
                    last = cut.iloc[-1]
                    v = round(float(last["value"]), 1)
                    a = float((rec.at - last["obs_at"]).total_seconds() // 3600)
            vitals[code] = (v, a)

        meta = ctx_map.loc[sub] if sub in ctx_map.index else None
        age = float(meta["age"]) if meta is not None and not pd.isna(meta["age"]) else 0.0
        sex = int(meta["sex"]) if meta is not None else -1

        row = Row(pid=sub, hour=hour, score=float(rec.score), label=0, age=age,
                  sex=sex, vitals=vitals, onset=onset.get(sub),
                  stay_septic=sub in onset, first_alert=first_alert.get(sub))

        at_iso = _iso(rec.at)
        art = f"art.{eid}"
        ev = {
            **base("verification_event"),
            "event_id": eid, "trace_id": trace, "idempotency_key": f"{sub}:{hour}",
            "system_id": C.SYSTEM_ID, "deployment_id": C.DEPLOYMENT_ID,
            "policy_hint": {"policy_id": policy["policy_id"],
                            "policy_version": policy["policy_version"]},
            "occurred_at": at_iso, "received_at": at_iso,
            "mode": d["mode"], "workflow_type": "w2.deterioration_prediction",
            "subject": {"subject_ref": f"sub.{sub}"},
            "context": {"site_id": C.SITE, "care_setting": "inpatient",
                        "population_tags": ["adult" if age >= 18 else "non_adult"],
                        "window_start": at_iso, "window_end": at_iso},
            "artifacts": [
                {"artifact_id": f"{art}.score", "role": "ai_output",
                 "modality": "model_score", "availability": "available",
                 "observed_at": at_iso, "subject_ref": f"sub.{sub}",
                 "source_system": C.SYSTEM_ID, "data_classification": "limited",
                 "inline_payload": {"score": round(float(rec.score), 6),
                                    "threshold": threshold,
                                    "alert": bool(rec.score >= threshold)}},
                {"artifact_id": f"{art}.input", "role": "source_input",
                 "modality": "structured", "availability": "available",
                 "observed_at": at_iso, "subject_ref": f"sub.{sub}",
                 "source_system": "hospital_export", "data_classification": "limited",
                 "inline_payload": {k: {"value": v[0], "age_hours": v[1]}
                                    for k, v in vitals.items()}},
                {"artifact_id": f"{art}.context", "role": "context",
                 "modality": "structured", "availability": "available",
                 "observed_at": at_iso, "subject_ref": f"sub.{sub}",
                 "source_system": "hospital_export", "data_classification": "limited",
                 "inline_payload": {"age": age, "sex": sex, "icu_hour": hour}},
                {"artifact_id": f"{art}.outcome", "role": "outcome",
                 "modality": "structured",
                 "availability": ("available" if "outcomes" in m.tables
                                  else m.declared.get("outcomes", "not_collected")),
                 "observed_at": at_iso, "subject_ref": f"sub.{sub}",
                 "source_system": "hospital_export",
                 "data_classification": "limited"},
            ],
            "lineage": [
                {"component_id": "cmp.subject-system", "component_type": "model",
                 "version": card["version"], "maker": d.get("maker", "vendor")},
                {"component_id": "cmp.intake", "component_type": "adapter",
                 "version": "1.0.0", "maker": "GoodHart"},
                {"component_id": "cmp.verifier", "component_type": "verifier",
                 "version": C.VERIFIER_VERSION, "maker": "GoodHart"},
            ],
        }
        crs = C.run_checks(row, threshold, ppv, eid, trace)
        for c in crs:
            c["started_at"] = c["completed_at"] = at_iso
        vd = C.compose(crs, policy, eid, trace, hour)
        vd["issued_at"] = at_iso

        keep(ev, eid)
        for c in crs:
            keep(c, c["check_result_id"])
        keep(vd, vd["verdict_id"])
        events.append(ev); checks.extend(crs); verdicts.append(vd)

        route = vd["required_disposition"]
        disps.append({**base("disposition_event"),
                      "disposition_event_id": f"dp.{eid}.1",
                      "verdict_id": vd["verdict_id"], "event_id": eid,
                      "state": "delivered", "occurred_at": at_iso,
                      "actor": {"actor_type": "verifier",
                                "actor_ref": "goodhart-router", "role": "system"},
                      "reason": f"{route['action']} to {route['target']}",
                      "sla_met": True})
        if route["action"] == "record":
            disps.append({**base("disposition_event"),
                          "disposition_event_id": f"dp.{eid}.2",
                          "verdict_id": vd["verdict_id"], "event_id": eid,
                          "state": "closed", "occurred_at": at_iso,
                          "actor": {"actor_type": "system",
                                    "actor_ref": "verification-ledger"},
                          "reason": "recorded", "sla_met": True})

        if "outcomes" not in m.tables:
            continue
        alerting_now = bool(rec.score >= threshold)
        tc = ("tc.alert-outcome" if vd["verdict"] == "flag" and alerting_now
              else "tc.input-condition" if vd["verdict"] == "flag"
              else "tc.pass-audit" if vd["verdict"] == "pass" else None)
        if tc is None:
            continue
        if tc == "tc.pass-audit" and rng.random() >= 0.25:
            continue

        last_hour = int(s[s.subject == sub]["hour"].max())
        if tc == "tc.input-condition":
            ages = [a for (_, a) in vitals.values() if a is not None]
            state, strength = "confirmed", "deterministic_authority"
            notes = (f"re-derived: oldest required observation {max(ages):.0f}h, "
                     f"limit {C.MAX_AGE_HOURS}h" if ages else "re-derived from input")
            resolved = at_iso
        elif hour + RESOLUTION_HOURS > last_hour and row.onset is None:
            state, strength, resolved = "unresolved", "none", None
            notes = f"window matures at hour {hour + RESOLUTION_HOURS}, data ends {last_hour}"
        elif tc == "tc.alert-outcome":
            if row.onset is None:
                state, strength, resolved = "overturned", "reference_label", at_iso
                notes = "no adjudicated outcome for this subject"
            elif row.first_alert is not None and row.first_alert < row.onset:
                state, strength, resolved = "confirmed", "reference_label", at_iso
                notes = (f"first alert hour {row.first_alert}, outcome {row.onset}, "
                         f"lead {row.onset - row.first_alert}h")
            else:
                state, strength, resolved = "overturned", "reference_label", at_iso
                notes = f"first alert hour {row.first_alert}, outcome {row.onset}"
        else:
            if row.onset is not None and row.first_alert is None:
                state, strength, resolved = "overturned", "reference_label", at_iso
                notes = f"outcome hour {row.onset}, no alert for this subject"
            else:
                state, strength, resolved = "confirmed", "reference_label", at_iso
                notes = "no outcome within the window"

        truths.append({**base("truth_resolution"),
                       "truth_resolution_id": f"tr.{eid}",
                       "verdict_id": vd["verdict_id"], "event_id": eid,
                       "truth_contract_id": tc, "state": state,
                       "source_strength": strength,
                       "source_artifact_ids": [f"{art}.outcome"],
                       "evaluated_at": at_iso, "resolved_at": resolved,
                       "adjudicator": {"actor_type": "system",
                                       "actor_ref": "hospital-outcome-export",
                                       "role": "reference_standard"},
                       "notes": notes})

    for x in disps:
        keep(x, x["disposition_event_id"])
    for t in truths:
        keep(t, t["truth_resolution_id"])
    for o in (*ccs, ab, fmp, policy):
        keep(o, o["object_type"])
    if invalid:
        raise SystemExit("contract validation failed:\n  "
                         + "\n  ".join(invalid[:10]))

    res = _write_api(m, out_dir, record_id, policy, ccs, ab, fmp, card,
                     events, checks, verdicts, disps, truths, threshold)
    keep(res["report"], "periodic_report")
    if invalid:
        raise SystemExit("contract validation failed:\n  "
                         + "\n  ".join(invalid[:10]))
    res["validated"] = validator is not None
    return res


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_api(mf, out_dir: Path, record_id, policy, ccs, ab, fmp, card,
               events, checks, verdicts, disps, truths, threshold) -> dict:
    api = Path(out_dir)
    (api / "events").mkdir(parents=True, exist_ok=True)
    for f in (api / "events").glob("*.json"):
        f.unlink()

    ck = {}
    for c in checks:
        ck.setdefault(c["event_id"], []).append(c)
    dp = {}
    for x in disps:
        dp.setdefault(x["event_id"], []).append(x)
    tr = {t["event_id"]: t for t in truths}
    vd = {v["event_id"]: v for v in verdicts}

    def w(name, obj):
        p = api / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
        return p.stat().st_size

    rows = []
    for ev in events:
        eid = ev["event_id"]
        v = vd[eid]
        t = tr.get(eid)
        payload = next(a["inline_payload"] for a in ev["artifacts"]
                       if a["role"] == "ai_output")
        w(f"events/{eid}.json",
          {"event": ev, "check_results": ck[eid], "verdict": v,
           "dispositions": dp.get(eid, []), "truth": t,
           "policy_ref": {"policy_id": policy["policy_id"],
                          "policy_version": policy["policy_version"]}})
        rows.append({
            "event_id": eid, "subject": ev["subject"]["subject_ref"][4:],
            "hour": next(a["inline_payload"]["icu_hour"] for a in ev["artifacts"]
                         if a["role"] == "context"),
            "at": ev["occurred_at"], "score": payload["score"],
            "alert": payload["alert"], "verdict": v["verdict"],
            "assessment": v["clinical_assessment"],
            "reason": v["metadata"]["reason_code"],
            "severity": v["metadata"]["worst_severity"],
            "latency_ms": v["metadata"]["latency_ms"], "sla_met": v["sla_met"],
            "checks": len(ck[eid]),
            "failed_checks": sum(1 for c in ck[eid] if c["status"] == "fail"),
            "truth": t["state"] if t else "not_sampled",
            "truth_contract": t["truth_contract_id"] if t else None,
            "disposition": dp.get(eid, [{}])[-1].get("state"),
            "tags": ev["context"]["population_tags"],
        })

    # Identical metric code to the pilot. If the demonstration is more careful
    # than the library, the customer is running the careless one.
    period_start = min(e["occurred_at"] for e in events) if events else now_iso()
    period_end = max(e["occurred_at"] for e in events) if events else period_start
    m = periodic.compute(events, verdicts, disps, truths, policy,
                         period_start, period_end)

    report = {
        **base("periodic_report"), "report_id": f"rep.{record_id}",
        "system_id": C.SYSTEM_ID, "deployment_ids": [C.DEPLOYMENT_ID],
        "policy_id": policy["policy_id"], "policy_version": policy["policy_version"],
        "period_start": period_start, "period_end": period_end,
        "coverage": m["coverage"], "confirmed_validity": m["confirmed_validity"],
        "landing": m["landing"], "evc": m["evc"], "truth_debt": m["truth_debt"],
        "changes": ["first run of this policy version"],
        "metadata": {**m["metadata"], "conditions": m["conditions"]},
        "findings": periodic.findings(m, extra=list(mf.notes)),
        "recommendation": m["recommendation"],
        "generated_at": now_iso(),
        "signed_by": [{"actor_type": "verifier", "actor_ref": "goodhart-monitor",
                       "role": f"verifier {C.VERIFIER_VERSION}"}],
    }

    w("onboarding.json", {"claim_contracts": ccs, "untestable_claims": [],
                          "authority_bundle": ab, "failure_mode_profile": fmp,
                          "vendor_card": card,
                          "record_ref": {"record": record_id, "sha256": "0" * 64}})
    w("policy.json", policy)
    w("report.json", report)
    w("index.json", {"system_id": C.SYSTEM_ID, "deployment_id": C.DEPLOYMENT_ID,
                     "site_id": C.SITE, "mode": policy["mode"],
                     "policy_id": policy["policy_id"],
                     "policy_version": policy["policy_version"],
                     "generated_from": Path(mf.path).name,
                     "contract_schema": "2.0.0", "threshold": threshold,
                     "rows": rows})
    return {"events": len(events), "coverage": m["coverage"],
            "validity": m["confirmed_validity"], "landing": m["landing"],
            "evc": m["evc"], "report": report, "api": str(api)}
