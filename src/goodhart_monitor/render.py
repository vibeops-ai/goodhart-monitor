"""Render a record for people. JSON is the artifact; this is the read.

A governance committee will not read JSON, and a record nobody reads is not
evidence. The markdown keeps the JSON's ordering and vocabulary exactly, so a
reader can move between them without translating: same section names, same
verdicts, LIMITS last but not smaller.
"""
from __future__ import annotations

from . import stats

MARK = {
    stats.HOLDS: "HOLDS",
    stats.FAILS: "FAILS",
    stats.INDETERMINATE: "INDETERMINATE",
    stats.NOT_APPLICABLE: "NOT APPLICABLE",
}


def _fmt(v) -> str:
    if v is None:
        return "not measurable"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "none"
    return str(v)


def _row(label: str, value) -> str:
    return f"| {label} | {_fmt(value)} |"


def _why(sec: dict) -> list[str]:
    """The sentence that justifies the verdict, never buried in the JSON only."""
    return ["", f"**Why this verdict** · {sec['why']}"] if sec.get("why") else []


def to_markdown(rec: dict) -> str:
    s = rec["sections"]
    subj = rec["subject"]
    out: list[str] = []
    A = out.append

    A(f"# Verification record {rec['record']}")
    A("")
    A(f"**Subject** · {subj['name']} {subj.get('version','')}  ")
    A(f"**Deployment population** · {rec['deployment_population']}  ")
    A(f"**Stream** · {rec['stream']['rows']:,} rows · "
      f"{rec['stream']['entities']:,} entities · "
      f"outcome prevalence {rec['stream']['prevalence_rows']:.4f}  ")
    A(f"**Record hash** · `{rec['record_sha256'][:32]}`")
    A("")
    h = rec["headline"]
    A(f"**Overall · {MARK.get(h['overall'], h['overall'])}**"
      + (f". Failing sections: {', '.join(h['sections_failing'])}"
         if h["sections_failing"] else ""))
    A("")
    A("There is no PASS in this vocabulary. The good outcome names its scope.")
    A("")

    # ---- acceptance
    a = s["acceptance"]
    A(f"## ACCEPTANCE · {MARK.get(a['verdict'], a['verdict'])}")
    A(f"_{a['question']}_")
    A("")
    if a.get("card_claim"):
        A(f"Card says: **{a['card_claim']}**"
          + (f" (on {a['card_population']})" if a.get("card_population") else ""))
        A("")
        A("| | |"); A("|---|---|")
        A(_row("measured here", a.get("measured_auroc")))
        A(_row("95% interval (entity bootstrap)", a.get("ci95_entity_bootstrap")))
        A(_row("gap vs card", a.get("gap_vs_card")))
        A(_row("tolerance allowed", a.get("tolerance")))
        A(_row("rows / entities", f"{a['n_rows']:,} / {a['n_entities']:,}"))
    out.extend(_why(a))
    A("")

    # ---- work
    w = s["work"]
    A(f"## WORK · {MARK.get(w['verdict'], w['verdict'])}")
    A(f"_{w['question']}_")
    A("")
    if w.get("card_claim"):
        A(f"Card says: **{w['card_claim']}**")
        A("")
    if w.get("threshold") is not None:
        A("| | |"); A("|---|---|")
        A(_row("threshold", w.get("threshold")))
        A(_row("alerts per 100 entity-days", w.get("alerts_per_100_entity_days")))
        A(_row("share of entities ever alerted",
               w.get("share_of_entities_ever_alerted")))
        A(_row("alert precision (row level)", w.get("row_level_ppv")))
        A(_row("entities evaluated per actionable catch",
               w.get("entities_evaluated_per_actionable_catch")))
        A(_row("…at sensitivity", w.get("sensitivity_actionable")))
        A("")
        A(f"> {w['note']}")
    out.extend(_why(w))
    A("")

    # ---- timing
    t = s["timing"]
    A(f"## TIMING · {MARK.get(t['verdict'], t['verdict'])}")
    A(f"_{t['question']}_")
    A("")
    if t.get("card_claim"):
        A(f"Card says: **{t['card_claim']}**")
        A("")
    if t.get("caught_at_all") is not None and t["verdict"] != stats.NOT_APPLICABLE:
        A("| | |"); A("|---|---|")
        A(_row("positive entities", t.get("positives")))
        A(_row("alerted on at all", t.get("caught_at_all")))
        A(_row("alerted before onset", t.get("caught_before_onset")))
        A(_row("share of catches after onset", t.get("share_of_catches_after_onset")))
        A(_row("median lead when early (h)", t.get("median_lead_hours_when_early")))
        A(_row(f"within the {_fmt(t.get('actionable_window_hours'))}h actionable window",
               t.get("caught_within_window")))
        A(_row("…as a share of catches", t.get("share_of_catches_within_window")))
        A("")
        if t.get("lead_time_distribution"):
            A("| lead time | catches |")
            A("|---|---:|")
            for b in t["lead_time_distribution"]:
                A(f"| {b['bucket']} | {b['catches']} |")
            A("")
        A(f"> {t['note']}")
    out.extend(_why(t))
    A("")

    # ---- drift
    d = s["drift"]
    A(f"## DRIFT · {MARK.get(d['verdict'], d['verdict'])}")
    A(f"_{d['question']}_")
    A("")
    A(f"Baseline: {d['baseline']} "
      f"(AUROC {_fmt(d.get('baseline_auroc'))}, PPV {_fmt(d.get('baseline_row_ppv'))}). "
      f"Review triggers at a {_fmt(d['thresholds']['auroc_drop_from_local_baseline'])} "
      f"AUROC drop or PPV below "
      f"{_fmt(d['thresholds']['ppv_floor_fraction_of_local'])} of baseline.")
    A("")
    if d.get("windows"):
        A("| window | entities | AUROC | PPV | alerts | review |")
        A("|---|---:|---:|---:|---:|---|")
        for wd in d["windows"]:
            flag = ", ".join(wd["review_reasons"]) if wd["review"] else (
                "underpowered" if wd["underpowered"] else "no")
            A(f"| {wd['window']} | {wd['entities']} | {_fmt(wd['auroc'])} | "
              f"{_fmt(wd['row_ppv'])} | {wd['alerts']} | {flag} |")
        A("")
        A(f"**{d['windows_triggering_review']} of {len(d['windows'])} windows "
          f"trigger review.** Ordering: {d.get('ordering','')}")
        A("")
        if d.get("note"):
            A(f"> {d['note']}")
    out.extend(_why(d))
    A("")

    # ---- subgroups
    g = s["subgroups"]
    A("## SUBGROUPS")
    A(f"_{g['question']}_")
    A("")
    if g["groups"]:
        A("| dimension | group | entities | prevalence | AUROC |")
        A("|---|---|---:|---:|---:|")
        for r in g["groups"]:
            A(f"| {r['dimension']} | {r['group']} | {r['n_entities']:,} | "
              f"{_fmt(r['prevalence'])} | {_fmt(r['auroc'])}"
              f"{' (underpowered)' if r['underpowered'] else ''} |")
        A("")
        A(f"AUROC spread across groups: "
          f"{_fmt(g.get('auroc_spread_across_groups'))}")
        A("")
    if g["dimensions_the_stream_cannot_answer"]:
        A(f"**Cannot be asked of this stream:** "
          f"{', '.join(g['dimensions_the_stream_cannot_answer'])}. {g['note']}")
    A("")

    # ---- unverifiable
    u = s["unverifiable_claims"]
    if u["claims"]:
        A("## CLAIMS THAT CANNOT BE TESTED")
        A("")
        for c in u["claims"]:
            A(f"- **{c['id']}** {c['text']}"
              + (f". _{c['detail']}_" if c.get("detail") else ""))
        A("")
        A(f"> {u['note']}")
        A("")

    # ---- limits, same weight
    A("## LIMITS")
    A("")
    A("Carried at the same weight as the findings.")
    A("")
    for item in s["limits"]["items"]:
        A(f"- {item}")
    A("")
    A("---")
    A(f"Re-run with the same inputs and this record hashes to "
      f"`{rec['record_sha256'][:32]}` again. If it does not, something changed "
      f"and the difference is the finding.")
    return "\n".join(out) + "\n"
