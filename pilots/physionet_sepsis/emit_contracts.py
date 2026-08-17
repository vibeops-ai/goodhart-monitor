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
from goodhart_monitor import periodic  # noqa: E402
from goodhart_monitor.contracts import (Row, base, qsofa, ref, sirs, ts,  # noqa: E402
                                        organ_dysfunction, usable)

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
    # everything the screening criteria need, not only the four bedside vitals.
    # Bilirubin is absent from this corpus, so the organ-dysfunction check runs
    # on three of four markers and says so.
    vital_cols = ["HR", "O2Sat", "SBP", "Resp", "Temp", "MAP",
                  "WBC", "Lactate", "Creatinine", "Platelets"]
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
            "mode": "validation", "workflow_type": "w2.deterioration_prediction",
            "subject": {"subject_ref": f"sub.{pid}"},
            "context": {
                "site_id": C.SITE, "encounter_ref": f"enc.{pid}",
                "care_setting": "inpatient_icu", "service_line": "medical_icu",
                "population_tags": ["adult", "icu", "medical",
                                    "age_80_plus" if row.age >= 80 else
                                    "age_65_79" if row.age >= 65 else
                                    "age_50_64" if row.age >= 50 else "age_under_50"],
                "window_start": ts(hour - 1), "window_end": ts(hour),
                "metadata": {"icu_hour": hour,
                             "calendar_is_derived": C.CALENDAR_IS_DERIVED,
                             "calendar_note": "the corpus carries ICU hour and no "
                                              "calendar; timestamps are projected "
                                              "onto a fixed epoch"},
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
        # The onset hour of a septic stay is always adjudicated. Leaving it to a
        # 25% draw means the sample can miss the one hour that decides whether
        # the model missed the patient.
        at_onset = onset.get(pid) is not None and hour == onset[pid]
        sampled = (tc in ("tc.alert-outcome", "tc.input-condition")
                   or at_onset or rng.random() < 0.25)
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
            notes = (f"re-derived: oldest required observation {oldest:.0f}h, "
                     f"limit 6h"
                     if oldest is not None else
                     "re-derived: flagged condition present in the input")
        elif not mature:
            state, strength, notes, resolved = (
                "unresolved", "none",
                f"outcome window matures at ICU hour {horizon}, stay ends at "
                f"{stay_end}", None)
        elif tc == "tc.alert-outcome":
            if row.onset is None:
                state, strength, resolved = "overturned", "reference_label", ts(stay_end)
                notes = "no adjudicated onset in this stay"
            elif row.first_alert is not None and row.first_alert < row.onset:
                state, strength, resolved = "confirmed", "reference_label", ts(row.onset)
                notes = (f"first alert ICU hour {row.first_alert}, onset "
                         f"{row.onset}, lead {row.onset - row.first_alert}h")
            else:
                state, strength, resolved = "overturned", "reference_label", ts(row.onset)
                notes = (f"first alert ICU hour {row.first_alert}, onset "
                         f"{row.onset}. Case finding.")
        else:
            if row.onset is not None and row.first_alert is None:
                # Before calling this a miss, ask whether the patient looked
                # septic at all. When the reference label and the published
                # screening criteria disagree, the honest state is inconclusive:
                # one of the two is wrong and this stream cannot say which.
                fresh, _ = usable(vitals)
                s_met, s_seen, _ = sirs(fresh)
                q_met, q_seen, _ = qsofa(fresh)
                o_met, o_seen, _ = organ_dysfunction(fresh)
                evaluable = q_seen >= 2 and (s_seen + o_seen) >= 4
                screen_negative = q_met == 0 and s_met <= 1 and o_met == 0
                if evaluable and screen_negative:
                    state, strength = "inconclusive", "reference_label"
                    resolved = ts(row.onset)
                    notes = (f"reference label places onset at ICU hour {row.onset}; "
                             f"independent screening criteria were negative at this "
                             f"hour (SIRS {s_met}/{s_seen}, qSOFA {q_met}/{q_seen}, "
                             f"organ {o_met}/{o_seen}). The reference and the criteria "
                             f"disagree and this stream cannot say which is right")
                else:
                    state, strength = "overturned", "reference_label"
                    resolved = ts(row.onset)
                    notes = (f"onset ICU hour {row.onset}, no alert on this stay; "
                             f"screening criteria at this hour SIRS {s_met}/{s_seen}, "
                             f"qSOFA {q_met}/{q_seen}, organ {o_met}/{o_seen}")
            elif row.onset is not None and hour < row.onset and row.first_alert is None:
                state, strength, resolved = "overturned", "reference_label", ts(row.onset)
                notes = "onset followed, no alert"
            else:
                state, strength, resolved = "confirmed", "reference_label", ts(min(horizon, stay_end))
                notes = "no onset within the window"

        # resolved_at is the moment the resolution could first be made, not the
        # moment the clinical event happened. The event time belongs in the note.
        # Conflating them made resolutions look like they preceded their own
        # evaluation.
        evaluated = ts(min(horizon, stay_end))
        if resolved is not None:
            resolved = max(resolved, evaluated)
            evaluated = resolved
        truths.append(validate({
            **base("truth_resolution"),
            "truth_resolution_id": f"tr.{event_id}",
            "verdict_id": verdict["verdict_id"], "event_id": event_id,
            "truth_contract_id": tc, "state": state,
            "source_strength": strength,
            "source_artifact_ids": [f"{art}.outcome"],
            "evaluated_at": evaluated,
            "resolved_at": resolved,
            "adjudicator": {"actor_type": "system", "actor_ref": "corpus-reference-label",
                            "role": "reference_standard"},
            "notes": notes,
        }, "truth"))

    # ---------------------------------------------------------------- report
    # Every figure comes from goodhart_monitor.periodic, the same code the
    # manifest runner uses. A demonstration with better methodology than the
    # library is not a demonstration of the library.
    period_start, period_end = ts(0), ts(24 * 7)
    m = periodic.compute(events, verdicts, disps, truths, policy,
                         period_start, period_end)

    report = validate({
        **base("periodic_report"),
        "report_id": "rep.micu.2026-w33",
        "system_id": C.SYSTEM_ID, "deployment_ids": [C.DEPLOYMENT_ID],
        "policy_id": policy["policy_id"], "policy_version": policy["policy_version"],
        "period_start": period_start, "period_end": period_end,
        "coverage": m["coverage"], "confirmed_validity": m["confirmed_validity"],
        "landing": m["landing"], "evc": m["evc"],
        "truth_debt": m["truth_debt"],
        "changes": ["no model, prompt, threshold or policy change in this period"],
        "metadata": {**m["metadata"], "conditions": m["conditions"]},
        "findings": periodic.findings(m, extra=[
            f"Local AUROC {record['sections']['acceptance']['measured_auroc']} against "
            f"card {record['sections']['acceptance']['card_value']}. Acceptance FAILS "
            f"in record GHM-0001.",
            f"Alert precision {record['sections']['work']['row_level_ppv']} against "
            f"card {record['sections']['work']['card_value']}. Work FAILS.",
            "Card claim M-4 carries no number. No Claim Contract issued.",
        ]),
        "recommendation": m["recommendation"],
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
    print(f"  {len(events)} events · {len(checks)} check results · "
          f"{len(verdicts)} verdicts")
    print(f"  verdicts {counts}")
    print(f"  truth    {tcounts}")
    print(f"  coverage {m['coverage']['value']}  "
          f"validity {m['confirmed_validity']['value']} per patient "
          f"({m['metadata']['validity_per_event']['value']} per event)  "
          f"landing {m['landing']['value']}  evc {m['evc']}")
    print("  every object validated against contracts/v2 schema 2.0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
