# GoodHart Monitor

Independent verification records for deployed clinical prediction models.

The product reads one thing: a **scored stream**. Whatever the model is, the
hospital already logs a score per patient (or per patient-hour) and eventually
learns the outcome. That is the whole input. The model stays the vendor's, its
weights stay wherever they are, and the record is produced from the output the
hospital already owns.

    goodhart-monitor selftest --dir /tmp/ghm      # whole loop, synthetic data
    goodhart-monitor intake  --manifest ours.toml # can we run on your export?
    goodhart-monitor run     --manifest ours.toml # verify it
    goodhart-monitor verify  --stream s.parquet --card card.json  # single record

See RUNBOOK.md for the four steps a hospital follows, including the five files
to export and what each one unlocks.

Exit code 0 no finding in scope · 1 a section FAILS · 2 something is
INDETERMINATE · 3 the inputs are not verifiable.

## Why the input is the output stream and not the model

On the structured-data path the scores are already in the EHR, so there is
nothing to negotiate with the vendor. The committee's own question is whether a
number holds on their population, measured with their data. A checker that needs
the vendor's artifact is a checker the vendor can decline. This one cannot be
declined, because the hospital already has everything it consumes.

## The input contract

Three columns are required, and the checker refuses loudly rather than coerce:

| column | meaning |
|---|---|
| `entity_id` | patient, encounter, stay: whatever the alert is about |
| `score` | what the model emitted, any monotone scale |
| `label` | the adjudicated outcome, 0/1 |

Two are optional and unlock sections when present:

| column | unlocks |
|---|---|
| `t` | TIMING and per-hour alert burden |
| any other column | SUBGROUPS, if named in the config |

Onset is inferred only when the label varies within an entity. A label that is
constant per entity is an entity-level outcome (readmission), and TIMING
returns NOT_APPLICABLE rather than inventing a lead time.

## The record

Five sections, each a question a committee already asks:

| Section | The question |
|---|---|
| ACCEPTANCE | does the card's headline number hold on this population? |
| WORK | what work does the alert stream create, and is that work valuable? |
| TIMING | does it warn before the event, or notice it afterwards? |
| DRIFT | windowed performance against explicit review thresholds |
| SUBGROUPS | where is the number worse, and which dimensions cannot be asked? |

Verdicts are HOLDS / FAILS / INDETERMINATE / NOT_APPLICABLE. **There is no
PASS.** The good outcome names its scope. LIMITS is carried at the same weight
as the findings, unverifiable claims are recorded rather than scored, and the
whole record is content-addressed: same inputs, same bytes, verifiable years
later by someone who was not in the room.

Every threshold that produces a verdict lives in `governance.toml`, is owned by
the hospital, and is printed on the record. A reader who disagrees with a
verdict can disagree with the policy rather than reverse-engineer it.

## Layout

    src/goodhart_monitor/    the product
      contract.py            the scored-stream contract; refuses rather than coerces
      card.py                vendor claims, typed
      config.py              governance thresholds, owned by the hospital
      stats.py               metrics, cluster bootstrap, verdict boundaries
      sections/              acceptance · work · timing · drift · subgroups
      record.py              assemble, canonicalise, hash
      render.py              the read; JSON stays the artifact
      cli.py                 verify · validate
    pilots/physionet_sepsis/ a demonstration, not the product
    tests/                   84 tests, including the two miscounts that shipped in v0

## Corpus and licence

The pilot runs on PhysioNet/CinC 2019, Open Access. See CORPUS.md for what is
recorded, what is derived, and how to reproduce the record. Code is MIT.

## The pilot in this repo

`pilots/physionet_sepsis/` builds a maker and checks it, end to end, on real
data: PhysioNet/CinC 2019, 40,336 real ICU stays across two real hospital
systems, Open Access. The maker trains on hospital A and ships a model card
whose every number is true. The checker runs on hospital B, which the maker
has never seen.

    ./pilots/physionet_sepsis/run.sh

The finding: a card with no false numbers on it still fails locally. That gap
is the product.

`to_stream.py` is the only step that touches the model, and it is on the pilot
side of the wall on purpose. A real hospital exports its score log instead; the
rest of the pipeline does not change.

## Honesty rails

Real data only. Anything constructed is labelled on the record: the PhysioNet
corpus carries no calendar, so DRIFT returns INDETERMINATE and says exactly
what the hospital would have to supply to make it answerable, rather than
dressing between-patient variation up as deterioration.
