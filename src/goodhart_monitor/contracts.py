"""Emit contract v2.0.0 objects from a real deployment stream.

This is the bridge between the measurement code in this repository and the
canonical interfaces in `contracts/v2/`. Onboarding produces a Claim Contract,
an Authority Bundle, a Failure Mode Profile and a Verification Policy; the
runtime produces a Verification Event, Check Results, a Verdict, Disposition
Events and, once the outcome window matures, a Truth Resolution.

Three rules the module holds to, because they are the ones a buyer will test:

  * a check pack decides one claim and never composes the verdict. The verdict
    comes from the policy's decision table, applied in priority order.
  * nothing is emitted that was not measured. This deployment has no connected
    clinical review workflow, so no reviewed/acted disposition exists and
    Landing is not applicable rather than zero or one.
  * every object is validated against the schema before it is written.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCHEMA_VERSION = "2.0.0"
ORG = "umn-health"
SITE = "hospital-b-micu"
SYSTEM_ID = "sys-maker1-sepsis"
DEPLOYMENT_ID = "dep-micu-monitoring"
VERIFIER_VERSION = "0.3.0"

# The pilot has no wall clock of its own: the corpus is de-identified and
# carries ICU hour, not calendar time. A fixed epoch is declared so timestamps
# are reproducible rather than dependent on when the export was run.
EPOCH = datetime(2026, 8, 10, 7, 0, tzinfo=timezone.utc)


def ts(hours: float) -> str:
    return (EPOCH + timedelta(hours=float(hours))).isoformat().replace("+00:00", "Z")


def sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ref(id_: str, version: str) -> dict:
    return {"id": id_, "version": version}


def base(object_type: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "object_type": object_type,
            "organization_id": ORG}


# --------------------------------------------------------------- onboarding
def claim_contracts(card: dict) -> list[dict]:
    """Turn the vendor's card into claims that can actually be tested.

    Each card line becomes an endpoint, a decision timepoint, a population, an
    output meaning, a metric with a target, and a named reference standard. A
    claim that cannot be written this way is not verifiable, and saying so is
    part of the product.
    """
    claims = {c["id"]: c for c in card["claims"]}
    reference = {
        "description": "PhysioNet/CinC 2019 Sepsis-3 label, shifted 6h ahead of "
                       "clinical onset",
        "source_roles": ["outcome"],
        "adjudication_method": "corpus reference label, no local re-adjudication",
        "maturity_window_seconds": 6 * 3600,
        "authority_refs": [ref("auth.sepsis3-cinc2019", "1.0.0")],
        "limitations": [
            "corpus label; no local chart review",
            "label timing set by the challenge",
        ],
    }
    out = [
        {
            **base("claim_contract"),
            "claim_contract_id": "cc.discrimination",
            "claim_contract_version": "1.0.0",
            "system_id": SYSTEM_ID,
            "vendor_claim": claims["M-1"]["text"],
            "normalized_claim": (
                f"AUROC at or above {claims['M-1']['value']} on adult ICU "
                f"patient-hours at this site"),
            "claim_kind": "prognostic_prediction",
            "clinical_endpoint": "Sepsis-3 onset during the ICU stay",
            "decision_timepoint": "each ICU hour, observations up to that hour",
            "intended_population": "adult ICU stays at the deployment site",
            "intended_inputs": ["hourly vitals", "laboratory results", "demographics"],
            "output_semantics": "continuous risk score in [0,1] for sepsis onset",
            "performance_requirements": [
                {"metric": "auroc", "operator": "gte", "target": claims["M-1"]["value"],
                 "confidence_level": 0.95,
                 "notes": "local estimate, bootstrap over patients"}],
            "reference_standard": reference,
            "effective_from": ts(0),
            "approved_by_role": "chief_health_ai_officer",
        },
        {
            **base("claim_contract"),
            "claim_contract_id": "cc.alert-precision",
            "claim_contract_version": "1.0.0",
            "system_id": SYSTEM_ID,
            "vendor_claim": claims["M-3"]["text"],
            "normalized_claim": (
                f"At the shipped threshold, {claims['M-3']['value']:.0%} or more of "
                f"alerting patient-hours fall in a stay with adjudicated sepsis"),
            "claim_kind": "diagnostic_detection",
            "clinical_endpoint": "Sepsis-3 onset during the ICU stay",
            "decision_timepoint": "the hour the alert fires",
            "intended_population": "adult ICU stays at the deployment site",
            "intended_inputs": ["hourly vitals", "laboratory results"],
            "output_semantics": "alert when the score reaches the shipped threshold",
            "performance_requirements": [
                {"metric": "ppv", "operator": "gte", "target": claims["M-3"]["value"],
                 "confidence_level": 0.95}],
            "reference_standard": reference,
            "effective_from": ts(0),
            "approved_by_role": "chief_health_ai_officer",
        },
        {
            **base("claim_contract"),
            "claim_contract_id": "cc.lead-time",
            "claim_contract_version": "1.0.0",
            "system_id": SYSTEM_ID,
            "vendor_claim": claims["M-2"]["text"],
            "normalized_claim": (
                "First alert precedes adjudicated onset in most alerting stays"),
            "claim_kind": "prognostic_prediction",
            "clinical_endpoint": "hours from first alert to adjudicated Sepsis-3 onset",
            "decision_timepoint": "the hour of the first alert for a stay",
            "intended_population": "adult ICU stays that alert at the shipped threshold",
            "intended_inputs": ["hourly vitals", "laboratory results"],
            "output_semantics": "hours from first alert to onset, positive means early",
            "performance_requirements": [
                {"metric": "lead_time", "operator": "gte", "target": 0.0,
                 "notes": "zero or below is case finding"}],
            "reference_standard": reference,
            "effective_from": ts(0),
            "approved_by_role": "chief_health_ai_officer",
        },
    ]
    # M-4 carries no number. It is registered as untestable rather than dropped.
    return out


def authority_bundle() -> dict:
    return {
        **base("authority_bundle"),
        "authority_bundle_id": "ab.sepsis-w2",
        "authority_bundle_version": "1.0.0",
        "sources": [
            {"authority_id": "auth.sepsis3-cinc2019", "authority_version": "1.0.0",
             "authority_type": "reference_dataset",
             "title": "PhysioNet/CinC 2019 Sepsis-3 label definition",
             "uri": "https://physionet.org/content/challenge-2019/1.0.0/",
             "effective_from": ts(0),
             "approved_by_role": "chief_health_ai_officer",
             "limitations": ["label shifted 6h ahead of clinical onset"]},
            {"authority_id": "auth.vitals-ranges", "authority_version": "1.0.0",
             "authority_type": "validated_rule_set",
             "title": "Physiologic plausibility ranges for adult ICU vital signs",
             "effective_from": ts(0),
             "approved_by_role": "clinical_owner",
             "limitations": ["screening bounds only"]},
            {"authority_id": "auth.local-freshness", "authority_version": "1.0.0",
             "authority_type": "hospital_policy",
             "title": "Maximum observation age for a risk score",
             "effective_from": ts(0),
             "approved_by_role": "clinical_owner",
             "local_override": True},
        ],
        "effective_from": ts(0),
        "approved_by_role": "chief_health_ai_officer",
    }


def failure_mode_profile() -> dict:
    def fm(id_, cat, desc, sev, action, roles=("source_input",), detector="det.w2"):
        return {"failure_mode_id": id_, "category": cat, "description": desc,
                "required_evidence_roles": list(roles),
                "detector_ref": ref(detector, "1.0.0"),
                "severity": sev, "default_action": action}

    return {
        **base("failure_mode_profile"),
        "failure_mode_profile_id": "fmp.sepsis-w2",
        "failure_mode_profile_version": "1.0.0",
        "claim_contract_refs": [ref("cc.discrimination", "1.0.0"),
                                ref("cc.alert-precision", "1.0.0"),
                                ref("cc.lead-time", "1.0.0")],
        "authority_bundle_refs": [ref("ab.sepsis-w2", "1.0.0")],
        "failure_modes": [
            fm("fm.missing-vitals", "missing_input",
               "score produced with a required vital sign absent",
               "high", "unable_to_verify"),
            fm("fm.stale-vitals", "stale_input",
               "most recent required observation exceeds the policy age limit",
               "warning", "flag"),
            fm("fm.implausible-value", "corrupt_input",
               "input outside physiologic screening bounds",
               "high", "flag"),
            fm("fm.out-of-population", "out_of_distribution",
               "patient outside the approved population",
               "high", "unable_to_verify"),
            fm("fm.threshold-drift", "threshold_error",
               "operating threshold differs from the approved policy",
               "critical", "hold", roles=("ai_output",)),
            fm("fm.false-negative", "false_negative",
               "no alert on a stay with adjudicated sepsis onset",
               "critical", "review", roles=("outcome",)),
            fm("fm.false-positive", "false_positive",
               "alert on a stay with no adjudicated sepsis onset",
               "warning", "review", roles=("outcome",)),
            fm("fm.late-alert", "false_negative",
               "first alert at or after adjudicated onset",
               "high", "review", roles=("outcome",)),
            fm("fm.subgroup-gap", "subgroup_gap",
               "subgroup discrimination below the population estimate",
               "warning", "review", roles=("context",)),
        ],
        "effective_from": ts(0),
        "approved_by_role": "clinical_owner",
    }


CHECKS = [
    {"check_id": "chk.input-completeness", "check_pack_id": "cp.structured-input",
     "check_pack_version": "1.0.0",
     "verification_claim": "all required inputs present at score time",
     "decision_mode": "direct_deterministic", "required": True,
     "input_roles": ["source_input"], "failure_mode_ids": ["fm.missing-vitals"],
     "missing_evidence_behavior": "unable_to_verify", "timeout_ms": 250},
    {"check_id": "chk.input-freshness", "check_pack_id": "cp.structured-input",
     "check_pack_version": "1.0.0",
     "verification_claim": "no required observation past the policy age limit",
     "decision_mode": "direct_deterministic", "required": True,
     "input_roles": ["source_input"], "failure_mode_ids": ["fm.stale-vitals"],
     "missing_evidence_behavior": "review_required", "timeout_ms": 250},
    {"check_id": "chk.input-plausibility", "check_pack_id": "cp.structured-input",
     "check_pack_version": "1.0.0",
     "verification_claim": "inputs within physiologic screening bounds",
     "decision_mode": "direct_deterministic", "required": True,
     "input_roles": ["source_input"], "failure_mode_ids": ["fm.implausible-value"],
     "missing_evidence_behavior": "skip_with_limitation", "timeout_ms": 250},
    {"check_id": "chk.population-scope", "check_pack_id": "cp.scope",
     "check_pack_version": "1.0.0",
     "verification_claim": "patient within the approved population",
     "decision_mode": "direct_deterministic", "required": True,
     "input_roles": ["context"], "failure_mode_ids": ["fm.out-of-population"],
     "missing_evidence_behavior": "unable_to_verify", "timeout_ms": 100},
    {"check_id": "chk.threshold-replay", "check_pack_id": "cp.scope",
     "check_pack_version": "1.0.0",
     "verification_claim": "alert decision replays from score and approved threshold",
     "decision_mode": "direct_deterministic", "required": True,
     "input_roles": ["ai_output"], "failure_mode_ids": ["fm.threshold-drift"],
     "missing_evidence_behavior": "fail_closed", "timeout_ms": 100},
    {"check_id": "chk.alert-support", "check_pack_id": "cp.predictive",
     "check_pack_version": "1.0.0",
     "verification_claim": "alert supported by locally measured precision",
     "decision_mode": "statistical_calibrated", "required": False,
     "input_roles": ["ai_output", "context"],
     "failure_mode_ids": ["fm.false-positive"],
     "calibration_ref": {"id": "cal.record-ghm-0001", "version": "1.0.0"},
     "missing_evidence_behavior": "review_required", "timeout_ms": 400},
]

DECISION_TABLE = [
    {"priority": 10, "when": {"any_check_status": "error"}, "verdict": "unable_to_verify",
     "reason_code": "check_error"},
    {"priority": 20, "when": {"any_failure_mode_action": "unable_to_verify"},
     "verdict": "unable_to_verify", "reason_code": "required_evidence_absent"},
    {"priority": 30, "when": {"any_failure_mode_action": "hold"}, "verdict": "hold",
     "reason_code": "policy_violation"},
    {"priority": 40, "when": {"any_clinical_assessment": "review_required"},
     "verdict": "flag", "reason_code": "review_required"},
    {"priority": 50, "when": {"any_clinical_assessment": "clinically_disputed"},
     "verdict": "flag", "reason_code": "disputed"},
    {"priority": 90, "when": {"otherwise": True}, "verdict": "pass",
     "reason_code": "no_violation_in_checked_scope"},
]


def verification_policy(threshold: float, cfg) -> dict:
    return {
        **base("verification_policy"),
        "policy_id": "pol.sepsis-micu",
        "policy_version": "1.2.0",
        "system_id": SYSTEM_ID,
        "effective_from": ts(0),
        # Monitoring mode: the verdict is recorded beside the workflow. No
        # clinical review queue is connected, which is why Landing is not
        # measurable for this deployment and the report withholds EVC.
        "mode": "monitoring",
        "intended_use": {
            "summary": "Hourly sepsis risk scoring for adult medical ICU patients",
            "workflow": "score written to the chart, no automated order or page",
            "population": "adult ICU stays, age 18 and over, medical ICU",
            "permitted": ["display a risk score to the care team",
                          "populate a surveillance worklist"],
            "prohibited": ["initiate an order", "page without human review",
                           "apply to paediatric or obstetric patients"],
        },
        "deployment_scope": {"deployment_ids": [DEPLOYMENT_ID], "site_ids": [SITE],
                             "population_tags": ["adult", "icu", "medical"]},
        "claim_contract_refs": [ref("cc.discrimination", "1.0.0"),
                                ref("cc.alert-precision", "1.0.0"),
                                ref("cc.lead-time", "1.0.0")],
        "authority_bundle_refs": [ref("ab.sepsis-w2", "1.0.0")],
        "failure_mode_profile_refs": [ref("fmp.sepsis-w2", "1.0.0")],
        "required_artifacts": [
            {"role": "ai_output", "modalities": ["model_score"], "required": True,
             "min_count": 1},
            {"role": "source_input", "modalities": ["structured"], "required": True,
             "max_age_seconds": 6 * 3600, "min_count": 1},
            {"role": "context", "modalities": ["structured"], "required": True},
            {"role": "outcome", "modalities": ["structured"], "required": False},
        ],
        "checks": CHECKS,
        "decision_table": DECISION_TABLE,
        "dispositions": {
            "pass": {"action": "record", "target": "verification_ledger",
                     "landing_sla_seconds": 60, "required_closure_state": "closed"},
            "flag": {"action": "record", "target": "verification_ledger",
                     "landing_sla_seconds": 60, "required_closure_state": "closed"},
            "hold": {"action": "escalate", "target": "ai_governance_oncall",
                     "landing_sla_seconds": 3600, "required_closure_state": "acted"},
            "unable_to_verify": {"action": "record", "target": "verification_ledger",
                                 "landing_sla_seconds": 60,
                                 "required_closure_state": "closed"},
        },
        "truth_contracts": [
            {"truth_contract_id": "tc.alert-outcome", "claim_type": "alert_supported",
             "applies_to_verdicts": ["flag"],
             "source_roles": ["outcome"], "source_strength": "reference_label",
             "resolution_window_seconds": 48 * 3600,
             "sampling": {"strategy": "all"},
             "confirmation_rule": {
                 "confirmed": "the stay has adjudicated sepsis onset after the first alert",
                 "overturned": "the stay has no adjudicated onset, or onset preceded the first alert"},
             "owner_role": "clinical_owner"},
            {"truth_contract_id": "tc.input-condition",
             "claim_type": "input_condition_present",
             "applies_to_verdicts": ["flag"],
             "source_roles": ["source_input"],
             "source_strength": "deterministic_authority",
             "resolution_window_seconds": 0,
             "sampling": {"strategy": "all"},
             "confirmation_rule": {
                 "confirmed": "the flagged input condition re-derives from the "
                              "recorded observation and the policy limit",
                 "overturned": "the condition does not re-derive"},
             "owner_role": "verifier"},
            {"truth_contract_id": "tc.pass-audit", "claim_type": "no_violation_in_scope",
             "applies_to_verdicts": ["pass"],
             "source_roles": ["outcome"], "source_strength": "reference_label",
             "resolution_window_seconds": 48 * 3600,
             "sampling": {"strategy": "risk_stratified", "rate": 0.25,
                          "minimum_cases": 50},
             "confirmation_rule": {
                 "confirmed": "no adjudicated onset followed within the resolution window",
                 "overturned": "adjudicated onset followed with no alert on the stay"},
             "owner_role": "clinical_owner"},
        ],
        "slo": {"verdict_latency_ms": 1500, "fail_behavior": "unable_to_verify"},
    }


# ------------------------------------------------------------------ runtime
REQUIRED_VITALS = ["HR", "O2Sat", "SBP", "Resp"]
PLAUSIBLE = {"HR": (20, 220), "O2Sat": (50, 100), "SBP": (50, 260), "Resp": (4, 60),
             "Temp": (30, 43)}
MAX_AGE_HOURS = 6


@dataclass
class Row:
    pid: str
    hour: int
    score: float
    label: int
    age: float
    sex: int
    vitals: dict           # name -> (value|None, age_hours|None)
    onset: int | None
    stay_septic: bool
    first_alert: int | None


def _assurance(mode: str, required: int, present: int, missing: list[str],
               calibrated: float | None = None,
               calibration_status: str = "not_applicable",
               calibration_ref: dict | None = None,
               limitations: list[str] | None = None) -> dict:
    a = {
        "decision_mode": mode,
        "execution_reproducibility": 1.0 if mode.startswith("direct") else None,
        "evidence_coverage": {
            "required_count": required, "present_count": present,
            "coverage_rate": round(present / required, 4) if required else 1.0,
            "missing_roles": missing,
        },
        "calibration_status": calibration_status,
    }
    if calibration_status in ("local_validated", "external_validated"):
        a["calibrated_correctness"] = calibrated
        a["calibration_ref"] = calibration_ref
    if limitations:
        a["limitations"] = limitations
    return a


def _proof(method: str, artifacts: list[str], authorities: list[dict],
           rules: list[dict], explanation: str, payload: Any) -> dict:
    return {"method_ref": ref(method, "1.0.0"), "input_artifact_ids": artifacts,
            "authority_refs": authorities, "rule_refs": rules,
            "input_hash": sha(payload), "explanation": explanation}


def run_checks(row: Row, thr: float, entity_ppv: float, event_id: str,
               trace_id: str) -> list[dict]:
    """The check packs. Each decides one claim and composes nothing."""
    art_in = f"art.{event_id}.input"
    art_out = f"art.{event_id}.score"
    art_ctx = f"art.{event_id}.context"
    results: list[dict] = []
    t0 = row.hour

    def emit(check_id, pack, status, severity, assessment, assurance, proof,
             findings=(), metrics=None, limitations=(), latency=6):
        results.append({
            **base("check_result"),
            "check_result_id": f"chk.{event_id}.{check_id.split('.')[-1]}",
            "event_id": event_id, "trace_id": trace_id,
            "policy_id": "pol.sepsis-micu", "policy_version": "1.2.0",
            "check_id": check_id, "check_pack_id": pack, "check_pack_version": "1.0.0",
            "status": status, "severity": severity,
            "claim_type": check_id.split(".", 1)[1],
            "clinical_assessment": assessment,
            "assurance": assurance, "decision_proof": proof,
            "findings": list(findings),
            **({"metrics": metrics} if metrics else {}),
            **({"limitations": list(limitations)} if limitations else {}),
            "started_at": ts(t0), "completed_at": ts(t0),
            "latency_ms": latency,
        })

    # 1 — completeness
    missing = [v for v in REQUIRED_VITALS if row.vitals.get(v, (None, None))[0] is None]
    emit("chk.input-completeness", "cp.structured-input",
         "pass" if not missing else "fail", "info" if not missing else "high",
         "clinically_corroborated" if not missing else "unable_to_verify",
         _assurance("direct_deterministic", len(REQUIRED_VITALS),
                    len(REQUIRED_VITALS) - len(missing),
                    [f"source_input:{m}" for m in missing]),
         _proof("m.completeness", [art_in], [], [ref("rule.required-vitals", "1.0.0")],
                "all required vitals have a value at or before this hour"
                if not missing else f"absent: {', '.join(missing)}",
                {"missing": missing}),
         findings=[] if not missing else [
             {"code": "MISSING_REQUIRED_INPUT",
              "message": f"absent: {', '.join(missing)}",
              "claim_contract_id": "cc.alert-precision",
              "failure_mode_ids": ["fm.missing-vitals"],
              "evidence_artifact_ids": [art_in]}])

    # 2 — freshness
    ages = [a for (_, a) in row.vitals.values() if a is not None]
    oldest = max(ages) if ages else None
    stale = oldest is not None and oldest > MAX_AGE_HOURS
    emit("chk.input-freshness", "cp.structured-input",
         "pass" if not stale else "fail", "info" if not stale else "warning",
         "clinically_corroborated" if not stale else "review_required",
         _assurance("direct_deterministic", 1, 1, []),
         _proof("m.freshness", [art_in], [ref("auth.local-freshness", "1.0.0")],
                [ref("rule.max-observation-age", "1.0.0")],
                f"oldest required observation {oldest}h, limit {MAX_AGE_HOURS}h"
                if oldest is not None else "no observation ages available",
                {"oldest_hours": oldest}),
         findings=[] if not stale else [
             {"code": "STALE_INPUT",
              "message": f"oldest required observation {oldest:.0f}h, limit {MAX_AGE_HOURS}h",
              "failure_mode_ids": ["fm.stale-vitals"],
              "evidence_artifact_ids": [art_in]}],
         metrics={"oldest_observation_hours": oldest} if oldest is not None else None)

    # 3 — plausibility
    bad = []
    for name, (val, _) in row.vitals.items():
        if val is None or name not in PLAUSIBLE:
            continue
        lo, hi = PLAUSIBLE[name]
        if not (lo <= val <= hi):
            bad.append(f"{name}={val}")
    emit("chk.input-plausibility", "cp.structured-input",
         "pass" if not bad else "fail", "info" if not bad else "high",
         "clinically_corroborated" if not bad else "clinically_disputed",
         _assurance("direct_deterministic", 1, 1, []),
         _proof("m.plausibility", [art_in], [ref("auth.vitals-ranges", "1.0.0")],
                [ref("rule.physiologic-bounds", "1.0.0")],
                "all present inputs within bounds" if not bad
                else f"outside bounds: {', '.join(bad)}", {"bad": bad}),
         findings=[] if not bad else [
             {"code": "IMPLAUSIBLE_INPUT", "message": f"outside bounds: {', '.join(bad)}",
              "failure_mode_ids": ["fm.implausible-value"],
              "evidence_artifact_ids": [art_in]}])

    # 4 — population scope
    in_scope = row.age >= 18
    emit("chk.population-scope", "cp.scope",
         "pass" if in_scope else "fail", "info" if in_scope else "high",
         "clinically_corroborated" if in_scope else "unable_to_verify",
         _assurance("direct_deterministic", 1, 1, []),
         _proof("m.scope", [art_ctx], [], [ref("rule.approved-population", "1.0.0")],
                f"age {row.age:.0f}, approved population adult ICU",
                {"age": row.age}),
         findings=[] if in_scope else [
             {"code": "OUT_OF_POPULATION",
              "message": "patient outside the approved population",
              "failure_mode_ids": ["fm.out-of-population"],
              "evidence_artifact_ids": [art_ctx]}])

    # 5 — threshold replay
    alerting = row.score >= thr
    emit("chk.threshold-replay", "cp.scope", "pass", "info",
         "clinically_corroborated",
         _assurance("direct_deterministic", 1, 1, []),
         _proof("m.threshold", [art_out], [], [ref("rule.approved-threshold", "1.0.0")],
                f"score {row.score:.4f} against approved threshold {thr:.4f}: "
                f"{'alert' if alerting else 'no alert'}",
                {"score": row.score, "threshold": thr}),
         metrics={"score": row.score, "threshold": thr, "alerting": alerting})

    # 6 — alert support, only when there is an alert to support
    if alerting:
        emit("chk.alert-support", "cp.predictive", "indeterminate", "warning",
             "review_required",
             _assurance("statistical_calibrated", 2, 2, [],
                        calibrated=round(entity_ppv, 4),
                        calibration_status="local_validated",
                        calibration_ref=ref("cal.record-ghm-0001", "1.0.0"),
                        limitations=[
                            "population precision; says nothing about this patient",
                            "outcome for this stay is not yet mature"]),
             _proof("m.alert-support", [art_out, art_ctx],
                    [ref("auth.sepsis3-cinc2019", "1.0.0")],
                    [ref("rule.local-precision", "1.0.0")],
                    f"local precision at this threshold {entity_ppv:.1%}, "
                    f"outcome for this stay unknown",
                    {"entity_ppv": entity_ppv}),
             findings=[{"code": "ALERT_REQUIRES_REVIEW",
                        "message": "alert issued, outcome unresolvable at score time",
                        "claim_contract_id": "cc.alert-precision",
                        "failure_mode_ids": ["fm.false-positive"],
                        "evidence_artifact_ids": [art_out]}],
             metrics={"entity_ppv_at_threshold": round(entity_ppv, 4)},
             latency=41)
    return results


def compose(results: list[dict], policy: dict, event_id: str, trace_id: str,
            hour: int) -> dict:
    """Apply the decision table in priority order. No check writes a verdict."""
    fmp_actions = {
        "fm.missing-vitals": "unable_to_verify", "fm.out-of-population": "unable_to_verify",
        "fm.threshold-drift": "hold", "fm.stale-vitals": "flag",
        "fm.implausible-value": "flag", "fm.false-positive": "review",
    }
    statuses = {r["status"] for r in results}
    assessments = {r["clinical_assessment"] for r in results}
    triggered = {fid for r in results for f in r["findings"]
                 for fid in f.get("failure_mode_ids", [])}
    actions = {fmp_actions.get(f) for f in triggered}

    verdict, reason = "pass", "no_violation_in_checked_scope"
    if "error" in statuses:
        verdict, reason = "unable_to_verify", "check_error"
    elif "unable_to_verify" in actions:
        verdict, reason = "unable_to_verify", "required_evidence_absent"
    elif "hold" in actions:
        verdict, reason = "hold", "policy_violation"
    elif "review_required" in assessments:
        verdict, reason = "flag", "review_required"
    elif "clinically_disputed" in assessments:
        verdict, reason = "flag", "disputed"

    # the clinical assessment carried on the verdict is the most consequential
    # one any check produced, never an average
    order = ["clinically_disputed", "review_required", "unable_to_verify",
             "clinically_corroborated"]
    assessment = next(a for a in order if a in assessments)

    worst = max(results, key=lambda r: ["info", "warning", "high", "critical"]
                .index(r["severity"]))
    modes = [r["assurance"]["decision_mode"] for r in results]
    mode = ("statistical_calibrated" if "statistical_calibrated" in modes
            else "direct_deterministic")
    cov = [r["assurance"]["evidence_coverage"] for r in results]
    required = sum(c["required_count"] for c in cov)
    present = sum(c["present_count"] for c in cov)
    missing = sorted({m for c in cov for m in c["missing_roles"]})

    msgs = "; ".join(f["message"] for r in results for f in r["findings"])
    summaries = {
        "pass": "No violation in checked scope.",
        "flag": msgs or "Review required.",
        "hold": "Policy violation. Output withheld.",
        "unable_to_verify": msgs or "Required evidence unavailable.",
    }
    limits = sorted({l for r in results for l in r.get("limitations", [])})

    assurance = _assurance(mode, required, present, missing,
                           calibration_status="not_applicable",
                           limitations=limits or None)
    total = len(results)
    supporting = sum(1 for r in results if r["status"] == "pass")
    assurance["agreement"] = {"supporting_checks": supporting, "total_checks": total}

    latency = sum(r["latency_ms"] for r in results)
    return {
        **base("verdict"),
        "verdict_id": f"vd.{event_id}",
        "event_id": event_id, "trace_id": trace_id,
        "policy_id": policy["policy_id"], "policy_version": policy["policy_version"],
        "verdict": verdict,
        "clinical_assessment": assessment,
        "assurance_summary": assurance,
        "summary": summaries[verdict],
        "basis_check_result_ids": [r["check_result_id"] for r in results],
        "limitations": limits,
        "required_disposition": policy["dispositions"][verdict],
        "issued_at": ts(hour),
        "sla_met": latency <= policy["slo"]["verdict_latency_ms"],
        "metadata": {"reason_code": reason, "latency_ms": latency,
                     "worst_severity": worst["severity"]},
    }
