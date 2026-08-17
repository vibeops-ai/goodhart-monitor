"""Hospital intake: read a mapping manifest, load the export, report readiness.

A hospital does not start with FHIR. It starts with an analyst who can write a
SQL query and drop four or five files somewhere. This module reads a manifest
that names those files and their columns, loads them, and answers one question
before any verdict is produced:

    which checks can run on what you have, and which cannot, and why

Readiness is reported per artifact role and per check, with the reason a check
is unavailable. A check that cannot run is reported as unavailable. It is never
silently skipped.

The manifest is TOML. A worked example lives at examples/manifest.toml.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class IntakeError(ValueError):
    pass


# Artifact roles the runtime needs, and what each unlocks. Keys match the
# `required_artifacts` roles on the verification policy.
ROLES = {
    "ai_outputs": {
        "required": True,
        "role": "ai_output",
        "columns": ["subject_ref", "occurred_at", "score"],
        "optional": ["event_id", "encounter_ref", "threshold", "alert"],
        "unlocks": ["chk.threshold-replay", "chk.alert-support"],
    },
    "observations": {
        "required": True,
        "role": "source_input",
        "columns": ["subject_ref", "observed_at", "code", "value"],
        "optional": ["encounter_ref", "unit"],
        "unlocks": ["chk.input-completeness", "chk.input-freshness",
                    "chk.input-plausibility"],
    },
    "population_context": {
        "required": True,
        "role": "context",
        "columns": ["subject_ref"],
        "optional": ["age", "sex", "site", "service_line", "care_setting"],
        "unlocks": ["chk.population-scope"],
    },
    "outcomes": {
        "required": False,
        "role": "outcome",
        "columns": ["subject_ref", "outcome_at", "outcome"],
        "optional": ["encounter_ref"],
        "unlocks": ["truth resolution: confirmed / overturned"],
    },
    "actions": {
        "required": False,
        "role": "disposition_evidence",
        "columns": ["subject_ref", "occurred_at", "state"],
        "optional": ["actor", "reason", "event_ref"],
        "unlocks": ["landing: delivered / reviewed / acted / closed"],
    },
}

AVAILABILITY = ("available", "restricted", "vendor_controlled",
                "requires_approval", "not_collected", "not_applicable")


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    source: str
    mapped: dict[str, str]
    availability: str = "available"


@dataclass
class Manifest:
    path: Path
    deployment: dict
    tables: dict[str, Table] = field(default_factory=dict)
    declared: dict[str, str] = field(default_factory=dict)   # name -> availability
    codes: dict[str, str] = field(default_factory=dict)      # their code -> ours
    notes: list[str] = field(default_factory=list)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    raise IntakeError(f"{path.name}: use .csv, .tsv or .parquet")


def load(manifest_path: str | Path) -> Manifest:
    p = Path(manifest_path)
    if not p.exists():
        raise IntakeError(f"no manifest at {p}")
    raw = tomllib.loads(p.read_text())

    dep = raw.get("deployment")
    if not dep:
        raise IntakeError("manifest has no [deployment] block")
    for k in ("organization_id", "system_id", "deployment_id", "site_id", "mode"):
        if k not in dep:
            raise IntakeError(f"[deployment] is missing {k}")
    if dep["mode"] not in ("validation", "monitoring", "advisory", "interventional"):
        raise IntakeError(f"[deployment] mode {dep['mode']!r} is not a contract mode")

    m = Manifest(path=p, deployment=dep,
                 codes={str(k): str(v) for k, v in raw.get("codes", {}).items()},
                 notes=list(raw.get("notes", {}).get("limitations", [])))

    base = p.parent
    for name, spec in ROLES.items():
        block = raw.get(name)
        if block is None:
            m.declared[name] = "not_collected"
            continue

        availability = block.get("availability", "available")
        if availability not in AVAILABILITY:
            raise IntakeError(f"[{name}] availability {availability!r} is not a "
                              f"contract state; use one of {', '.join(AVAILABILITY)}")
        m.declared[name] = availability
        if availability != "available":
            continue

        if "path" not in block:
            raise IntakeError(f"[{name}] is available but names no path")
        fp = (base / block["path"]).resolve()
        if not fp.exists():
            raise IntakeError(f"[{name}] path does not exist: {fp}")

        cols = block.get("columns", {})
        df = _read(fp)
        missing_src = [src for src in cols.values()
                       if isinstance(src, str) and src not in df.columns]
        if missing_src:
            raise IntakeError(
                f"[{name}] {fp.name} has no column(s) {', '.join(missing_src)}. "
                f"Present: {', '.join(map(str, df.columns[:12]))}"
                + (" …" if len(df.columns) > 12 else ""))

        # constants are allowed in place of a column, which is how a hospital
        # supplies a fixed threshold or a single site id
        mapped: dict[str, str] = {}
        for target, src in cols.items():
            if isinstance(src, str) and src in df.columns:
                mapped[target] = src
            else:
                df[f"__const_{target}"] = src
                mapped[target] = f"__const_{target}"

        need = spec["columns"]
        absent = [c for c in need if c not in mapped]
        if absent:
            raise IntakeError(
                f"[{name}.columns] must map {', '.join(need)}; missing "
                f"{', '.join(absent)}")

        m.tables[name] = Table(name=name, df=df, source=str(fp), mapped=mapped,
                               availability=availability)
    return m


# --------------------------------------------------------------- readiness
@dataclass
class Readiness:
    manifest: Manifest
    roles: list[dict]
    checks: list[dict]
    blocking: list[str]

    @property
    def runnable(self) -> bool:
        return not self.blocking


def assess(m: Manifest) -> Readiness:
    roles, checks, blocking = [], [], []

    for name, spec in ROLES.items():
        state = m.declared.get(name, "not_collected")
        t = m.tables.get(name)
        rows = 0 if t is None else len(t.df)
        subjects = 0
        if t is not None and "subject_ref" in t.mapped:
            subjects = int(t.df[t.mapped["subject_ref"]].nunique())
        roles.append({
            "table": name, "artifact_role": spec["role"], "availability": state,
            "required": spec["required"], "rows": rows, "subjects": subjects,
            "unlocks": spec["unlocks"],
            "source": None if t is None else Path(t.source).name,
        })
        if spec["required"] and state != "available":
            blocking.append(f"{name} is {state}; the runtime cannot produce a "
                            f"verdict without it")

    obs = m.tables.get("observations")
    have_codes = set()
    if obs is not None:
        raw_codes = set(obs.df[obs.mapped["code"]].astype(str).unique())
        have_codes = {m.codes.get(c, c) for c in raw_codes}

    from .contracts import CHECKS, REQUIRED_VITALS, QSOFA_INPUTS, CLINICAL_INPUTS
    for c in CHECKS:
        why, ok = "", True
        needed_roles = c["input_roles"]
        for r in needed_roles:
            table = next((n for n, s in ROLES.items() if s["role"] == r), None)
            if table and m.declared.get(table) != "available":
                ok, why = False, f"{table} is {m.declared.get(table, 'not_collected')}"
        if ok and c["check_id"] == "chk.input-completeness":
            miss = [v for v in REQUIRED_VITALS if v not in have_codes]
            if miss:
                ok = False
                why = (f"observations carry no code mapped to {', '.join(miss)}; "
                       f"add them under [codes]")
        if ok and c["check_pack_id"] == "cp.clinical-deterioration":
            need = [n for n in (QSOFA_INPUTS if c["check_id"] == "chk.qsofa"
                                else CLINICAL_INPUTS)
                    if n not in have_codes and n != "GCS"]
            if c["check_id"] == "chk.independent-signal" and \
                    any(n not in have_codes for n in ("Resp", "SBP")):
                ok = False
                why = ("qSOFA needs respiratory rate and systolic pressure; map "
                       "them under [codes]")
            elif need:
                why = f"runs on partial evidence; unmapped: {', '.join(need[:5])}"
        if ok and c["check_id"] == "chk.population-scope":
            ctx = m.tables.get("population_context")
            if ctx is None or "age" not in ctx.mapped:
                ok, why = False, "population_context maps no age column"
        if ok and c["check_id"] == "chk.alert-support":
            if m.tables.get("outcomes") is None:
                why = ("runs, but its calibration is unavailable until an outcome "
                       "export exists")
        checks.append({"check_id": c["check_id"], "available": ok,
                       "decision_mode": c["decision_mode"],
                       "required": c["required"], "why": why})
        if not ok and c["required"]:
            blocking.append(f"{c['check_id']} is required by the policy and cannot "
                            f"run: {why}")

    if m.tables.get("outcomes") is None:
        checks.append({"check_id": "truth.resolution", "available": False,
                       "decision_mode": "n/a", "required": False,
                       "why": "no outcome export; every verdict stays unresolved "
                              "and Confirmed Validity has no denominator"})
    if m.tables.get("actions") is None:
        checks.append({"check_id": "landing.closure", "available": False,
                       "decision_mode": "n/a", "required": False,
                       "why": "no action export; Landing has no denominator and "
                              "EVC is withheld"})
    return Readiness(manifest=m, roles=roles, checks=checks, blocking=blocking)


def render(r: Readiness) -> str:
    d = r.manifest.deployment
    out = [
        f"intake  {r.manifest.path.name}",
        f"  organization {d['organization_id']}  system {d['system_id']}",
        f"  deployment   {d['deployment_id']}  site {d['site_id']}  mode {d['mode']}",
        "",
        "  table                availability      rows   subjects  artifact role",
    ]
    for x in r.roles:
        out.append(f"  {x['table']:<20} {x['availability']:<16} "
                   f"{x['rows']:>7,} {x['subjects']:>10,}  {x['artifact_role']}")
    out += ["", "  check                     runs  mode                    note"]
    for c in r.checks:
        mark = "yes " if c["available"] else "no  "
        out.append(f"  {c['check_id']:<25} {mark}  {c['decision_mode']:<22}  "
                   f"{c['why']}")
    out.append("")
    if r.blocking:
        out.append("  BLOCKING")
        for b in r.blocking:
            out.append(f"    {b}")
        out.append("")
        out.append("  no verdict will be produced until these are resolved")
    else:
        out.append("  ready: every required check can run")
        degraded = [c for c in r.checks if not c["available"]]
        if degraded:
            out.append(f"  {len(degraded)} optional capability(ies) unavailable; "
                       f"the periodic report will say so")
    return "\n".join(out)


# ------------------------------------------------------------------ synthetic
def synthesise(dest: Path, n_subjects: int = 60, hours: int = 36,
               seed: int = 7, with_outcomes: bool = True) -> Path:
    """Write an export in the required shape, with no real data in it.

    A hospital can install the verifier, run this, and see the whole loop work
    before anyone opens a data access request.
    """
    rng = np.random.default_rng(seed)
    dest.mkdir(parents=True, exist_ok=True)

    scores, obs, ctx, outs = [], [], [], []
    t0 = pd.Timestamp("2026-01-05T07:00:00Z")
    for i in range(n_subjects):
        sid = f"SUBJ{i:04d}"
        septic = rng.random() < 0.25
        onset = int(rng.integers(8, hours - 2)) if septic else None
        age = int(rng.integers(21, 92))
        ctx.append({"mrn_hash": sid, "age_years": age,
                    "sex_code": int(rng.integers(0, 2)), "unit": "MICU"})
        for h in range(hours):
            at = t0 + pd.Timedelta(hours=h)
            risk = 0.05 + rng.normal(0, 0.02)
            if onset is not None and h >= onset - 6:
                risk += 0.35
            risk = float(np.clip(risk, 0.001, 0.999))
            scores.append({"score_id": f"{sid}-{h:03d}", "mrn_hash": sid,
                           "score_time": at.isoformat().replace("+00:00", "Z"),
                           "sepsis_risk": round(risk, 6)})
            sick = onset is not None and h >= onset - 6
            # bedside vitals, hourly
            for code, lo, hi in (("HEART_RATE", 95 if sick else 60, 125),
                                 ("SPO2", 90, 100),
                                 ("SYS_BP", 85 if sick else 105, 140),
                                 ("RESP_RATE", 20 if sick else 12, 26),
                                 ("TEMP_C", 38.0 if sick else 36.2, 38.9),
                                 ("MAP_MMHG", 58 if sick else 70, 95)):
                if rng.random() < 0.9:
                    obs.append({"mrn_hash": sid,
                                "obs_time": at.isoformat().replace("+00:00", "Z"),
                                "obs_code": code,
                                "obs_value": round(float(rng.uniform(lo, hi)), 1)})
            # labs, episodic, so the age-out path is exercised too
            if h % 8 == 0:
                for code, lo, hi in (("WBC_K", 13 if sick else 5, 17),
                                     ("LACTATE_MMOL", 2.2 if sick else 0.7, 3.4),
                                     ("CREAT_MGDL", 0.6, 1.4),
                                     ("PLT_K", 90 if sick else 180, 320)):
                    obs.append({"mrn_hash": sid,
                                "obs_time": at.isoformat().replace("+00:00", "Z"),
                                "obs_code": code,
                                "obs_value": round(float(rng.uniform(lo, hi)), 1)})
        if with_outcomes and onset is not None:
            at = t0 + pd.Timedelta(hours=onset)
            outs.append({"mrn_hash": sid,
                         "event_time": at.isoformat().replace("+00:00", "Z"),
                         "event_name": "SEPSIS3"})

    pd.DataFrame(scores).to_csv(dest / "ai_outputs.csv", index=False)
    pd.DataFrame(obs).to_csv(dest / "observations.csv", index=False)
    pd.DataFrame(ctx).to_csv(dest / "population_context.csv", index=False)
    if with_outcomes:
        pd.DataFrame(outs).to_csv(dest / "outcomes.csv", index=False)

    manifest = f'''# Generated by `goodhart-monitor selftest`. Synthetic data, no PHI.
[deployment]
organization_id = "example-health"
system_id       = "sys-example-sepsis"
deployment_id   = "dep-selftest"
site_id         = "micu"
mode            = "monitoring"

[ai_outputs]
path = "ai_outputs.csv"
[ai_outputs.columns]
event_id     = "score_id"
subject_ref  = "mrn_hash"
occurred_at  = "score_time"
score        = "sepsis_risk"
threshold    = 0.32          # a constant here; a column name also works

[observations]
path = "observations.csv"
[observations.columns]
subject_ref = "mrn_hash"
observed_at = "obs_time"
code        = "obs_code"
value       = "obs_value"

[population_context]
path = "population_context.csv"
[population_context.columns]
subject_ref = "mrn_hash"
age         = "age_years"
sex         = "sex_code"
site        = "unit"

{'[outcomes]' if with_outcomes else '# [outcomes] not exported'}
{'path = "outcomes.csv"' if with_outcomes else '# availability = "requires_approval"'}
{'[outcomes.columns]' if with_outcomes else ''}
{'subject_ref = "mrn_hash"' if with_outcomes else ''}
{'outcome_at  = "event_time"' if with_outcomes else ''}
{'outcome     = "event_name"' if with_outcomes else ''}

# No clinician action export exists in this environment.
[actions]
availability = "not_collected"

# Their code -> the code the check packs expect
[codes]
HEART_RATE   = "HR"
SPO2         = "O2Sat"
SYS_BP       = "SBP"
RESP_RATE    = "Resp"
TEMP_C       = "Temp"
MAP_MMHG     = "MAP"
WBC_K        = "WBC"
LACTATE_MMOL = "Lactate"
CREAT_MGDL   = "Creatinine"
PLT_K        = "Platelets"

[notes]
limitations = ["synthetic data generated by selftest; no clinical meaning"]
'''
    mp = dest / "manifest.toml"
    mp.write_text(manifest)
    return mp
