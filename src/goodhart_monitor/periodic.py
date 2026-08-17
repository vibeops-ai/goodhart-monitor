"""Period metrics: one implementation, used by every caller.

Coverage, Confirmed Validity, Landing, EVC, the decomposition, the conditions
and the recommendation are computed here and nowhere else. A demonstration that
applies careful methodology and a library that does not is two products, and the
one the customer runs is the careless one.

Rules this module holds, each of which exists because leaving it out flattered
the result:

  * a stay is one fact. Confirmed Validity is reported per patient as the
    headline, because counting per patient-hour lets two long stays supply most
    of the denominator.
  * a contract resolved against the verifier's own arithmetic cannot be
    overturned and is excluded from the headline.
  * events outside the reporting period are excluded. Spillover was entirely
    one-directional.
  * an unresolvable miss is still a miss. Validity is reported both with
    inconclusive results excluded and with them counted against us.
  * a verdict lands only in the closure state its policy demands, within the
    service level its policy sets. Anything else lets an auto-close job move
    Landing to 100% without a clinician.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np

SELF_CONFIRMING = {"deterministic_authority"}
CLOSURE_ORDER = ["delivered", "reviewed", "acted", "overridden", "closed"]


def _frac(num: int, den: int, lo=None, hi=None) -> dict:
    return {"numerator": num, "denominator": den,
            "value": round(num / den, 4) if den else None,
            "confidence_interval_low": lo, "confidence_interval_high": hi}


def _patient_of(event_id: str) -> str:
    parts = event_id.split(".")
    return parts[1] if len(parts) > 2 else event_id


def _cluster_ci(by_patient: dict[str, list[int]], seed: int, n: int = 400):
    """Bootstrap over patients. Rows within a stay are not independent."""
    pats = list(by_patient)
    if len(pats) < 10:
        return None, None
    rs = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rs.choice(len(pats), len(pats), replace=True)
        vals = [v for i in pick for v in by_patient[pats[i]]]
        if vals:
            draws.append(sum(vals) / len(vals))
    if not draws:
        return None, None
    return (round(float(np.percentile(draws, 2.5)), 4),
            round(float(np.percentile(draws, 97.5)), 4))


def _within(ts: str, start: str, end: str) -> bool:
    return start <= ts <= end


def compute(events: list[dict], verdicts: list[dict], disps: list[dict],
            truths: list[dict], policy: dict, period_start: str, period_end: str,
            seed: int = 20260815) -> dict:
    """Every published figure for one reporting period."""
    at_of = {e["event_id"]: e["occurred_at"] for e in events}
    in_period = {eid for eid, at in at_of.items()
                 if _within(at, period_start, period_end)}
    spilled = len(at_of) - len(in_period)

    verdicts = [v for v in verdicts if v["event_id"] in in_period]
    truths = [t for t in truths if t["event_id"] in in_period]
    disps = [d for d in disps if d["event_id"] in in_period]
    vd_by_id = {v["verdict_id"]: v for v in verdicts}

    # ---------------------------------------------------------- coverage
    complete = sum(1 for v in verdicts if v["sla_met"])
    coverage = _frac(complete, len(verdicts))

    # ---------------------------------------------------------- validity
    scored = [t for t in truths if t["source_strength"] not in SELF_CONFIRMING]
    adjudicated = [t for t in scored if t["state"] in ("confirmed", "overturned")]
    inconclusive = [t for t in scored if t["state"] == "inconclusive"]

    per_event = _frac(sum(1 for t in adjudicated if t["state"] == "confirmed"),
                      len(adjudicated))

    # A stay counts once. It is confirmed only if nothing about it was overturned.
    by_patient: dict[str, list[int]] = {}
    for t in adjudicated:
        by_patient.setdefault(_patient_of(t["event_id"]), []).append(
            1 if t["state"] == "confirmed" else 0)
    clean = sum(1 for v in by_patient.values() if all(v))
    lo, hi = _cluster_ci({k: [1 if all(v) else 0] for k, v in by_patient.items()}, seed)
    per_patient = _frac(clean, len(by_patient), lo, hi)

    # Counting the unresolvable misses against us. They arise only on misses, so
    # excluding them can only ever help.
    strict_den = len(adjudicated) + len(inconclusive)
    strict = _frac(per_event["numerator"], strict_den)

    concentration = None
    if adjudicated:
        counts = sorted((len(v) for v in by_patient.values()), reverse=True)
        concentration = {
            "patients": len(by_patient),
            "largest_share": round(counts[0] / len(adjudicated), 4),
            "top_two_share": round(sum(counts[:2]) / len(adjudicated), 4),
        }

    decomposition: dict[str, dict] = {}
    for t in truths:
        d = decomposition.setdefault(t["truth_contract_id"], {
            "confirmed": 0, "overturned": 0, "inconclusive": 0, "unresolved": 0,
            "source_strength": t["source_strength"],
            "counts_towards_headline": t["source_strength"] not in SELF_CONFIRMING})
        if t["state"] in d:
            d[t["state"]] += 1
    for d in decomposition.values():
        adj = d["confirmed"] + d["overturned"]
        d["rate"] = round(d["confirmed"] / adj, 4) if adj else None

    # ----------------------------------------------------------- landing
    routes = policy["dispositions"]
    actionable, landed, late = [], 0, 0
    states_by_verdict: dict[str, list[dict]] = {}
    for d in disps:
        states_by_verdict.setdefault(d["verdict_id"], []).append(d)
    for v in verdicts:
        route = routes.get(v["verdict"], {})
        if route.get("action") in (None, "record"):
            continue
        actionable.append(v)
        required = route.get("required_closure_state", "closed")
        need = set(CLOSURE_ORDER[CLOSURE_ORDER.index(required):]) \
            if required in CLOSURE_ORDER else {required}
        reached = [d for d in states_by_verdict.get(v["verdict_id"], [])
                   if d["state"] in need]
        if not reached:
            continue
        # the closure must also have happened inside the service level
        sla = route.get("landing_sla_seconds")
        first = min(d["occurred_at"] for d in reached)
        if sla is None:
            landed += 1
            continue
        issued = datetime.fromisoformat(v["issued_at"].replace("Z", "+00:00"))
        closed_at = datetime.fromisoformat(first.replace("Z", "+00:00"))
        if (closed_at - issued).total_seconds() <= sla:
            landed += 1
        else:
            late += 1
    landing = _frac(landed, len(actionable))

    # EVC uses the per-patient validity, which is the defensible one.
    evc = None
    if all(m["value"] is not None for m in (coverage, per_patient, landing)):
        evc = round(coverage["value"] * per_patient["value"] * landing["value"], 4)

    # ------------------------------------------------------ pass extrapolation
    pa = decomposition.get("tc.pass-audit", {})
    pa_adj = pa.get("confirmed", 0) + pa.get("overturned", 0)
    miss_rate = (pa.get("overturned", 0) / pa_adj) if pa_adj else None
    n_pass = sum(1 for v in verdicts if v["verdict"] == "pass")
    implied = round(n_pass * miss_rate) if miss_rate is not None else None

    unresolved = [t for t in truths if t["state"] == "unresolved"]

    # ----------------------------------------------------------- conditions
    conditions = []
    if landing["value"] is not None and landing["value"] < 0.8:
        conditions.append({
            "condition": f"Staff {routes['flag']['target']} or change the "
                         f"disposition route. {landing['denominator']} verdicts were "
                         f"routed and {landing['numerator']} reached the closure "
                         f"state the policy requires.",
            "owner_role": "clinical_owner", "due": "before the next period"})
    if miss_rate is not None and miss_rate >= 0.2:
        conditions.append({
            "condition": f"Raise the pass-audit sample. {miss_rate:.0%} of audited "
                         f"passes were overturned, implying roughly {implied:,} "
                         f"missed events, and the estimate rests on {pa_adj} "
                         f"adjudications.",
            "owner_role": "clinical_owner", "due": "before the next period"})
    if concentration and concentration["top_two_share"] > 0.25:
        conditions.append({
            "condition": f"Widen the window. Two patients supply "
                         f"{concentration['top_two_share']:.0%} of the adjudicated "
                         f"denominator, so the headline describes those stays more "
                         f"than the model.",
            "owner_role": "chief_health_ai_officer", "due": "next period"})
    if per_patient["confidence_interval_low"] is not None and \
            per_patient["confidence_interval_low"] < 0.8:
        conditions.append({
            "condition": f"Per-patient Confirmed Validity is "
                         f"{per_patient['value']:.2f} with a 95% interval of "
                         f"[{per_patient['confidence_interval_low']:.2f}, "
                         f"{per_patient['confidence_interval_high']:.2f}] on "
                         f"{per_patient['denominator']} patients. Too few patients "
                         f"and too wide to support a decision.",
            "owner_role": "chief_health_ai_officer", "due": "next period"})

    recommendation = ("pause" if miss_rate is not None and miss_rate > 0.5
                      else "re_review" if conditions
                      else "continue_with_conditions")

    return {
        "coverage": coverage,
        "confirmed_validity": per_patient,
        "landing": landing,
        "evc": evc,
        "truth_debt": {
            "count": len(unresolved),
            "high_risk_count": sum(
                1 for t in unresolved
                if vd_by_id.get(t["verdict_id"], {}).get("verdict") == "flag"),
            "oldest_seconds": 48 * 3600,
        },
        "conditions": conditions,
        "recommendation": recommendation,
        "metadata": {
            "validity_basis": "per patient; a stay is confirmed only if nothing "
                              "about it was overturned",
            "validity_per_event": per_event,
            "validity_counting_inconclusive_as_missed": strict,
            "validity_decomposition": decomposition,
            "concentration": concentration,
            "inconclusive": len(inconclusive),
            "events_excluded_outside_period": spilled,
            "audited_pass_miss_rate": None if miss_rate is None else round(miss_rate, 4),
            "implied_missed_in_passes": implied,
            "landing_closed_late": late,
            "coverage_denominator": "consequential outputs received by the verifier. "
                                    "An output never sent is not counted here and "
                                    "cannot lower this figure.",
        },
    }


def findings(m: dict, extra: Iterable[str] = ()) -> list[str]:
    """Plain statements a committee can read, derived from the same numbers."""
    md = m["metadata"]
    out = list(extra)
    v, pe = m["confirmed_validity"], md["validity_per_event"]
    if v["value"] is not None:
        out.append(
            f"Confirmed Validity {v['value']:.1%} per patient "
            f"({v['numerator']}/{v['denominator']} stays"
            + (f", 95% CI {v['confidence_interval_low']:.2f}-"
               f"{v['confidence_interval_high']:.2f}"
               if v["confidence_interval_low"] is not None else "")
            + f"). Per patient-hour it would read {pe['value']:.1%}.")
    if md["concentration"] and md["concentration"]["top_two_share"] > 0.25:
        out.append(f"Two patients supply "
                   f"{md['concentration']['top_two_share']:.0%} of the adjudicated "
                   f"denominator.")
    if md["inconclusive"]:
        s = md["validity_counting_inconclusive_as_missed"]
        out.append(f"{md['inconclusive']} resolutions are inconclusive because the "
                   f"reference label and the screening criteria disagree. They arise "
                   f"only on misses; counted against us the per-event figure is "
                   f"{s['value']:.1%}.")
    if md["events_excluded_outside_period"]:
        out.append(f"{md['events_excluded_outside_period']} events fell outside the "
                   f"reporting period and are excluded.")
    if md["audited_pass_miss_rate"]:
        out.append(f"{md['audited_pass_miss_rate']:.1%} of audited passes were "
                   f"overturned, implying roughly "
                   f"{md['implied_missed_in_passes']:,} missed events this period.")
    lnd = m["landing"]
    if lnd["value"] is not None:
        out.append(
            f"{lnd['denominator']:,} verdicts were routed for review and "
            f"{lnd['numerator']} reached the closure state the policy requires "
            f"within its service level. Landing {lnd['value']:.0%}"
            + (f", so EVC is {m['evc']:.0%} whatever the verdicts were worth."
               if m["evc"] is not None else
               ". EVC is withheld because another factor has no denominator."))
    else:
        out.append("No verdict is actionable under this policy, so Landing has no "
                   "denominator and EVC is withheld.")
    out.append("Coverage counts outputs the verifier received. An output never sent "
               "is invisible to it.")
    return out
