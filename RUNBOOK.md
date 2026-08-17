# Runbook

How a hospital runs the verifier on its own data. Four steps, each of which can
stop and be reviewed before the next one starts.

## 1. Install and prove it works, with no data access

```
git clone https://github.com/vibeops-ai/goodhart-monitor
cd goodhart-monitor
uv venv && uv pip install -e '.[dev]'

.venv/bin/goodhart-monitor selftest --dir /tmp/ghm
```

`selftest` writes a synthetic export, reads it back through the same intake path
a real export uses, and runs the whole loop. It touches no hospital system and
contains no patient data. Expect:

```
verified 2,160 synthetic events
  coverage 1.0  validity 1.0  landing None  evc withheld
```

Landing is `None` because the synthetic export declares no clinician action
file. That is the correct answer and the same one a first real run gives.

To see validity degrade the same way:

```
.venv/bin/goodhart-monitor selftest --dir /tmp/ghm-no-outcomes --no-outcomes
```

Run the test suite if you want the internals checked too:

```
.venv/bin/python -m pytest
```

## 2. Export five files

Column names do not matter. The manifest maps yours onto ours. Timestamps must
be ISO 8601; if they carry no timezone they are read as UTC.

| File | Required | One row per | Must contain |
|---|---|---|---|
| `ai_outputs` | yes | model output | subject id, timestamp, score, threshold |
| `observations` | yes | observation | subject id, timestamp, code, value |
| `population_context` | yes | subject | subject id, age; site and sex if available |
| `outcomes` | no | adjudicated outcome | subject id, timestamp, outcome name |
| `actions` | no | clinician action | subject id, timestamp, state |

Subject ids can be hashed or pseudonymous, as long as the same id appears in
every file. The verifier never needs a name, an MRN or a date of birth.

Threshold may be a column or a constant in the manifest. One run covers one
threshold; a change of threshold is a new policy version and a new run.

Without `outcomes`, every verdict stays `unresolved` and Confirmed Validity has
no denominator. Without `actions`, Landing has none. In both cases EVC is
withheld and the periodic report names the missing factor. The run still
produces per-output verdicts, which is most of the value of a first month.

## 3. Write the manifest and check readiness

Copy the one `selftest` generated at `/tmp/ghm/manifest.toml` and edit it. Then:

```
.venv/bin/goodhart-monitor intake --manifest ours.toml
```

This produces the readiness report and exits non-zero if a required check
cannot run:

```
  table                availability      rows   subjects  artifact role
  ai_outputs           available        412,908     3,104  ai_output
  observations         available      1,880,441     3,104  source_input
  population_context   available          3,104     3,104  context
  outcomes             requires_approval      0         0  outcome
  actions              not_collected          0         0  disposition_evidence

  check                     runs  mode                    note
  chk.input-completeness    yes   direct_deterministic
  chk.input-freshness       yes   direct_deterministic
  ...
  truth.resolution          no    n/a    no outcome export; every verdict stays
                                         unresolved and Confirmed Validity has
                                         no denominator
```

Nothing is written and no verdict exists at this stage. This is the artifact to
take to a data governance meeting: it states exactly which files unlock which
checks.

## 4. Run

```
.venv/bin/goodhart-monitor run --manifest ours.toml --out out/api --record-id GHM-UMN-001
```

Every object is validated against `contracts/v2/goodhart-verifier-contracts.schema.json`
before it is written. A single invalid object stops the run.

Output is a static read API:

```
out/api/onboarding.json     claim contracts, authority bundle, failure modes
out/api/policy.json         the executable policy that produced the verdicts
out/api/index.json          one row per event
out/api/report.json         coverage, confirmed validity, landing, EVC, debt
out/api/events/<id>.json    the full record for one event
```

To read it in the console UI, serve the API and the site from the same origin:

```
cp -r out/api  ../goodhart-web/public/api/v1
cd ../goodhart-web && npm run build && npx astro preview
# http://localhost:4321/app/
```

## Where the data sits

Everything above runs on the hospital's own machine against the hospital's own
files. Nothing is uploaded. If a run has to happen inside your environment
rather than on a laptop, the same commands work in a container with the export
mounted read-only.

Institutional approval is a separate gate. Low technical friction does not
remove data access review, security review, or an IRB or QI determination where
one applies.

## Reproducing the published pilot

The PhysioNet demonstration in `pilots/physionet_sepsis/` is public and needs no
approval:

```
./pilots/physionet_sepsis/run.sh                       # verification record
.venv/bin/python pilots/physionet_sepsis/emit_contracts.py   # contract objects
```
