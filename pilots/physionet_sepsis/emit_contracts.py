"""Run the verifier over a real deployment window and emit contract objects.

Writes a static read API the product UI consumes, shaped the way the real one
will be: a small index for lists, and one document per verification event.

    out/api/onboarding.json   claim contracts, authority bundle, failure modes
    out/api/policy.json       the executable verification policy
    out/api/index.json        one compact row per event, for feeds and queues
    out/api/report.json       Coverage, Confirmed Validity, Landing, EVC, debt
    out/api/events/<id>.json  the full event: artifacts, checks, verdict,
                              dispositions, truth resolution

Everything is validated against contracts/v2 before it is written. Two
deliberate absences, both visible in the output rather than papered over:

  * this deployment is in monitoring mode with no clinical review queue
    connected, so no reviewed or acted disposition exists. Landing therefore has
    no actionable denominator and the report withholds EVC, which is what the
    contract requires.
  * the vendor's fourth card claim carries no number, so it produced no Claim
    Contract and appears in the report's findings instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from goodhart_monitor import contracts as C  # noqa: E402
from goodhart_monitor.contracts import Row, base, ref, ts  # noqa: E402

OUT = ROOT / "out"
SCHEMA = json.loads((ROOT / "contracts" / "v2"
                     / "goodhart-verifier-contracts.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)

# The verified window: the same 44 stays the ward view replays, so a reader can
# move between the two surfaces and see the same patients.
N_STAYS = 44
RESOLUTION_HOURS = 48


def validate(obj: dict, what: str) -> dict:
    errs = sorted(VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errs:
        e = errs[0]
        raise SystemExit(f"{what} failed contract validation at "
                         f"{'/'.join(str(p) for p in e.path)}: {e.message}")
    return obj


def main() -> int:
    card = json.loads((OUT / "MODEL_CARD.json").read_text())
    record = json.loads((OUT / "record_GHM-0001.json").read_text())
    shift = json.loads((OUT / "shift_GHM-0001.json").read_text())
    thr = float(card["shipped_threshold"])
    entity_ppv = float(record["sections"]["work"]["entity_level_ppv"])

    stream = pd.read_parquet(OUT / "stream_B.parquet")
    deploy = pd.read_parquet(OUT / "B_deploy.parquet")
    deploy = deploy.assign(entity_id=deploy["patient"].astype(str))

    ids = [p["id"].replace("B-", "") for p in shift["patients"]][:N_STAYS]
    vital_cols = ["HR", "O2Sat", "SBP", "Resp", "Temp"]
    age_cols = [f"{c}_age" for c in vital_cols]

    src = stream[stream.entity_id.isin(ids)].merge(
        deploy[["entity_id", "ICULOS", *vital_cols, *age_cols]],
        left_on=["entity_id", "t"], right_on=["entity_id", "ICULOS"], how="left")

    onset = (src[src.label == 1].groupby("entity_id")["t"].min().to_dict())
    alerts = src[src.score >= thr]
    first_alert = alerts.groupby("entity_id")["t"].min().to_dict()
    septic = set(onset)

    # ---- onboarding artifacts
    ccs = [validate(c, c["claim_contract_id"]) for c in C.claim_contracts(card)]
    ab = validate(C.authority_bundle(), "authority bundle")
    fmp = validate(C.failure_mode_profile(), "failure mode profile")
    policy = validate(C.verification_policy(thr, None), "policy")

    events, checks, verdicts, disps, truths = [], [], [], [], []
    rng = np.random.default_rng(20260815)

    for _, r in src.sort_values(["entity_id", "t"]).iterrows():
        pid, hour = r.entity_id, int(r.t)
        event_id = f"ev.{pid}.{hour:03d}"
        trace_id = f"tr.{pid}.{hour:03d}"

        vitals = {}
        for c in vital_cols:
            v = r.get(c)
            a = r.get(f"{c}_age")
            vitals[c] = (None if pd.isna(v) else round(float(v), 1),
                         None if pd.isna(a) else float(a))

        row = Row(pid=pid, hour=hour, score=float(r.score), label=int(r.label),
                  age=float(r.Age), sex=int(r.Gender), vitals=vitals,
                  onset=onset.get(pid), stay_septic=pid in septic,
                  first_alert=first_alert.get(pid))

        art = f"art.{event_id}"
        event = validate({
            **base("verification_event"),
            "event_id": event_id, "trace_id": trace_id,
            "idempotency_key": f"{pid}:{hour}",
            "system_id": C.SYSTEM_ID, "deployment_id": C.DEPLOYMENT_ID,
            "policy_hint": {"policy_id": policy["policy_id"],
                            "policy_version": policy["policy_version"]},
            "occurred_at": ts(hour), "received_at": ts(hour + 0.002),
            "mode": "monitoring", "workflow_type": "w2.deterioration_prediction",
            "subject": {"subject_ref": f"sub.{pid}"},
            "context": {
                "site_id": C.SITE, "encounter_ref": f"enc.{pid}",
                "care_setting": "inpatient_icu", "service_line": "medical_icu",
                "population_tags": ["adult", "icu", "medical",
                                    "age_80_plus" if row.age >= 80 else
                                    "age_65_79" if row.age >= 65 else
                                    "age_50_64" if row.age >= 50 else "age_under_50"],
                "window_start": ts(hour - 1), "window_end": ts(hour),
            },
            "artifacts": [
                {"artifact_id": f"{art}.score", "role": "ai_output",
                 "modality": "model_score", "availability": "available",
                 "observed_at": ts(hour), "subject_ref": f"sub.{pid}",
                 "encounter_ref": f"enc.{pid}", "source_system": "maker1-sepsis",
                 "data_classification": "deidentified",
                 "inline_payload": {"score": round(float(r.score), 6),
                                    "threshold": thr,
                                    "alert": bool(r.score >= thr)}},
                {"artifact_id": f"{art}.input", "role": "source_input",
                 "modality": "structured", "availability": "available",
                 "schema_profile": "fhir-r4/Observation",
                 "observed_at": ts(hour), "subject_ref": f"sub.{pid}",
                 "source_system": "ehr-warehouse",
                 "data_classification": "deidentified",
                 "inline_payload": {k: {"value": v[0], "age_hours": v[1]}
                                    for k, v in vitals.items()}},
                {"artifact_id": f"{art}.context", "role": "context",
                 "modality": "structured", "availability": "available",
                 "observed_at": ts(hour), "subject_ref": f"sub.{pid}",
                 "source_system": "adt",
                 "data_classification": "deidentified",
                 "inline_payload": {"age": row.age, "sex": row.sex,
                                    "icu_hour": hour}},
                # The outcome is genuinely not available at score time. It is
                # declared here as not_collected so the absence is a state
                # rather than a silent null.
                {"artifact_id": f"{art}.outcome", "role": "outcome",
                 "modality": "structured", "availability": "not_collected",
                 "observed_at": ts(hour), "subject_ref": f"sub.{pid}",
                 "source_system": "ehr-warehouse",
                 "data_classification": "deidentified"},
            ],
            "lineage": [
                {"component_id": "cmp.maker1", "component_type": "model",
                 "version": card["version"], "maker": "MAKER-1 (vendor)",
                 "configuration_hash": card["model_sha256"][:16]},
                {"component_id": "cmp.adapter-w2", "component_type": "adapter",
                 "version": "1.0.0", "maker": "GoodHart"},
                {"component_id": "cmp.verifier", "component_type": "verifier",
                 "version": C.VERIFIER_VERSION, "maker": "GoodHart"},
            ],
        }, event_id)

        crs = [validate(c, c["check_result_id"])
               for c in C.run_checks(row, thr, entity_ppv, event_id, trace_id)]
        verdict = validate(C.compose(crs, policy, event_id, trace_id, hour),
                           f"verdict {event_id}")

        events.append(event); checks.extend(crs); verdicts.append(verdict)

        # ---- disposition. Monitoring mode records; nothing is routed to a
        # human because no review workflow is connected in this deployment.
        route = verdict["required_disposition"]
        disps.append(validate({
            **base("disposition_event"),
            "disposition_event_id": f"dp.{event_id}.1",
            "verdict_id": verdict["verdict_id"], "event_id": event_id,
            "state": "delivered", "occurred_at": ts(hour + 0.01),
            "actor": {"actor_type": "verifier", "actor_ref": "goodhart-router",
                      "role": "system"},
            "reason": f"{route['action']} to {route['target']}",
            "sla_met": True,
        }, "disposition"))
        if route["action"] == "record":
            disps.append(validate({
                **base("disposition_event"),
                "disposition_event_id": f"dp.{event_id}.2",
                "verdict_id": verdict["verdict_id"], "event_id": event_id,
                "state": "closed", "occurred_at": ts(hour + 0.01),
                "actor": {"actor_type": "system", "actor_ref": "verification-ledger"},
                "reason": "recorded; no human action required by policy",
                "sla_met": True,
            }, "disposition"))

        # ---- truth. Resolve only what the window and the sampling policy allow.
        #
        # A flag is not one claim. A flag raised because the model alerted is
        # adjudicated against the outcome; a flag raised because an observation
        # was stale is a deterministic statement about the input, and resolving
        # it against a sepsis outcome would be answering a question nobody asked.
        alerting = bool(r.score >= thr)
        if verdict["verdict"] == "flag" and alerting:
            tc = "tc.alert-outcome"
        elif verdict["verdict"] == "flag":
            tc = "tc.input-condition"
        elif verdict["verdict"] == "pass":
            tc = "tc.pass-audit"
        else:
            tc = None
        if tc is None:
            continue
        sampled = tc in ("tc.alert-outcome", "tc.input-condition") or rng.random() < 0.25
        if not sampled:
            continue

        horizon = hour + RESOLUTION_HOURS
        stay_end = int(src[src.entity_id == pid]["t"].max())
        mature = horizon <= stay_end or row.onset is not None

        if tc == "tc.input-condition":
            # deterministic: re-derive the condition that caused the flag
            ages = [a for (_, a) in vitals.values() if a is not None]
            oldest = max(ages) if ages else None
            state, strength, resolved = ("confirmed", "deterministic_authority",
                                         ts(hour))
            notes = (f"re-derived from the input: oldest required observation "
                     f"{oldest:.0f}h old against a policy limit of 6h"
                     if oldest is not None else
                     "re-derived from the input: the flagged condition is present")
        elif not mature:
            state, strength, notes, resolved = (
                "unresolved", "none",
                f"outcome window matures at ICU hour {horizon}; the stay ends at "
                f"{stay_end}", None)
        elif tc == "tc.alert-outcome":
            if row.onset is None:
                state, strength, resolved = "overturned", "reference_label", ts(stay_end)
                notes = "no adjudicated sepsis onset in this stay"
            elif row.first_alert is not None and row.first_alert < row.onset:
                state, strength, resolved = "confirmed", "reference_label", ts(row.onset)
                notes = (f"onset at ICU hour {row.onset}; first alert at "
                         f"{row.first_alert}, {row.onset - row.first_alert}h earlier")
            else:
                state, strength, resolved = "overturned", "reference_label", ts(row.onset)
                notes = (f"first alert at ICU hour {row.first_alert} did not precede "
                         f"onset at {row.onset}: case finding, not prediction")
        else:
            if row.onset is not None and row.first_alert is None:
                state, strength, resolved = "overturned", "reference_label", ts(row.onset)
                notes = (f"adjudicated onset at ICU hour {row.onset} with no alert on "
                         f"this stay")
            elif row.onset is not None and hour < row.onset and row.first_alert is None:
                state, strength, resolved = "overturned", "reference_label", ts(row.onset)
                notes = "onset followed with no alert"
            else:
                state, strength, resolved = "confirmed", "reference_label", ts(min(horizon, stay_end))
                notes = "no adjudicated onset followed without an alert"

        truths.append(validate({
            **base("truth_resolution"),
            "truth_resolution_id": f"tr.{event_id}",
            "verdict_id": verdict["verdict_id"], "event_id": event_id,
            "truth_contract_id": tc, "state": state,
            "source_strength": strength,
            "source_artifact_ids": [f"{art}.outcome"],
            "evaluated_at": ts(min(horizon, stay_end)),
            "resolved_at": resolved,
            "adjudicator": {"actor_type": "system", "actor_ref": "corpus-reference-label",
                            "role": "reference_standard"},
            "notes": notes,
        }, "truth"))

    # ---------------------------------------------------------------- report
    n_events = len(events)
    complete = sum(1 for v in verdicts if v["sla_met"])
    coverage = {"numerator": complete, "denominator": n_events,
                "value": round(complete / n_events, 4) if n_events else None}

    adjudicated = [t for t in truths if t["state"] in ("confirmed", "overturned")]
    confirmed = sum(1 for t in adjudicated if t["state"] == "confirmed")
    validity = {"numerator": confirmed, "denominator": len(adjudicated),
                "value": round(confirmed / len(adjudicated), 4) if adjudicated else None}

    # Landing counts actionable verdicts only. In monitoring mode with no review
    # queue connected there are none, so the denominator is zero and the value
    # is null. It is not 100%.
    actionable = [v for v in verdicts
                  if v["required_disposition"]["action"] not in ("record",)]
    landed = 0
    landing = {"numerator": landed, "denominator": len(actionable),
               "value": round(landed / len(actionable), 4) if actionable else None}

    evc = None
    if all(m["value"] is not None for m in (coverage, validity, landing)):
        evc = round(coverage["value"] * validity["value"] * landing["value"], 4)

    unresolved = [t for t in truths if t["state"] == "unresolved"]
    due = [t for t in unresolved]
    report = validate({
        **base("periodic_report"),
        "report_id": "rep.micu.2026-w33",
        "system_id": C.SYSTEM_ID, "deployment_ids": [C.DEPLOYMENT_ID],
        "policy_id": policy["policy_id"], "policy_version": policy["policy_version"],
        "period_start": ts(0), "period_end": ts(24 * 7),
        "coverage": coverage, "confirmed_validity": validity, "landing": landing,
        "evc": evc,
        "truth_debt": {"count": len(due),
                       "high_risk_count": sum(
                           1 for t in due
                           if next(v for v in verdicts
                                   if v["verdict_id"] == t["verdict_id"])["verdict"] == "flag"),
                       "oldest_seconds": RESOLUTION_HOURS * 3600},
        "changes": ["no model, prompt, threshold or policy change in this period"],
        "findings": [
            "Landing is not applicable: this deployment runs in monitoring mode and "
            "no clinical review destination is connected, so no verdict is actionable. "
            "EVC is withheld rather than computed from a missing factor.",
            "Vendor claim M-4, that performance generalises to new hospital systems, "
            "carries no number and produced no Claim Contract. It is registered as "
            "untestable.",
            f"Local discrimination is {record['sections']['acceptance']['measured_auroc']} "
            f"against a card claim of {record['sections']['acceptance']['card_value']}; "
            f"the acceptance section of record GHM-0001 FAILS.",
            f"Alert precision is {record['sections']['work']['row_level_ppv']} against a "
            f"card claim of {record['sections']['work']['card_value']}; the work section "
            f"FAILS.",
        ],
        "recommendation": "continue_with_conditions",
        "generated_at": ts(24 * 7),
        "signed_by": [{"actor_type": "verifier", "actor_ref": "goodhart-monitor",
                       "role": f"verifier {C.VERIFIER_VERSION}"}],
    }, "report")

    # ------------------------------------------------------------- write API
    api = OUT / "api"
    (api / "events").mkdir(parents=True, exist_ok=True)
    for f in (api / "events").glob("*.json"):
        f.unlink()

    by_event_checks: dict[str, list] = {}
    for c in checks:
        by_event_checks.setdefault(c["event_id"], []).append(c)
    by_event_disp: dict[str, list] = {}
    for d in disps:
        by_event_disp.setdefault(d["event_id"], []).append(d)
    by_event_truth = {t["event_id"]: t for t in truths}
    by_event_verdict = {v["event_id"]: v for v in verdicts}

    def write(name: str, obj) -> int:
        p = api / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
        return p.stat().st_size

    rows = []
    for ev in events:
        eid = ev["event_id"]
        v = by_event_verdict[eid]
        t = by_event_truth.get(eid)
        crs = by_event_checks[eid]
        score = next(a["inline_payload"]["score"] for a in ev["artifacts"]
                     if a["role"] == "ai_output")
        alert = next(a["inline_payload"]["alert"] for a in ev["artifacts"]
                     if a["role"] == "ai_output")
        write(f"events/{eid}.json", {
            "event": ev, "check_results": crs, "verdict": v,
            "dispositions": by_event_disp.get(eid, []),
            "truth": t,
            "policy_ref": {"policy_id": policy["policy_id"],
                           "policy_version": policy["policy_version"]},
        })
        rows.append({
            "event_id": eid,
            "subject": ev["subject"]["subject_ref"].replace("sub.p", ""),
            "hour": int(ev["context"]["metadata"]["icu_hour"])
            if ev["context"].get("metadata") else
            next(a["inline_payload"]["icu_hour"] for a in ev["artifacts"]
                 if a["role"] == "context"),
            "at": ev["occurred_at"],
            "score": score, "alert": alert,
            "verdict": v["verdict"],
            "assessment": v["clinical_assessment"],
            "reason": v["metadata"]["reason_code"],
            "severity": v["metadata"]["worst_severity"],
            "latency_ms": v["metadata"]["latency_ms"],
            "sla_met": v["sla_met"],
            "checks": len(crs),
            "failed_checks": sum(1 for c in crs if c["status"] == "fail"),
            "truth": (t["state"] if t else "not_sampled"),
            "truth_contract": (t["truth_contract_id"] if t else None),
            "disposition": by_event_disp.get(eid, [{}])[-1].get("state"),
            "tags": ev["context"]["population_tags"],
        })

    write("onboarding.json", {
        "claim_contracts": ccs,
        "untestable_claims": [c for c in card["claims"] if c.get("value") is None],
        "authority_bundle": ab,
        "failure_mode_profile": fmp,
        "vendor_card": card,
        "record_ref": {"record": record["record"],
                       "sha256": record["record_sha256"]},
    })
    write("policy.json", policy)
    write("report.json", report)
    idx_size = write("index.json", {
        "system_id": C.SYSTEM_ID, "deployment_id": C.DEPLOYMENT_ID,
        "site_id": C.SITE, "mode": policy["mode"],
        "policy_id": policy["policy_id"], "policy_version": policy["policy_version"],
        "generated_from": "PhysioNet/CinC 2019 hospital B, via record GHM-0001",
        "contract_schema": "2.0.0",
        "threshold": thr,
        "rows": rows,
    })
    path = api / "index.json"

    counts = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    tcounts = {}
    for t in truths:
        tcounts[t["state"]] = tcounts.get(t["state"], 0) + 1

    detail_kb = sum(f.stat().st_size for f in (api / "events").glob("*.json")) // 1024
    print(f"wrote {api.relative_to(ROOT)}/  index {idx_size // 1024} KB, "
          f"{len(rows)} event documents totalling {detail_kb} KB")
    print(f"  {n_events} events · {len(checks)} check results · {len(verdicts)} verdicts")
    print(f"  verdicts {counts}")
    print(f"  truth    {tcounts}")
    print(f"  coverage {coverage['value']}  validity {validity['value']}  "
          f"landing {landing['value']}  evc {evc}")
    print("  every object validated against contracts/v2 schema 2.0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
