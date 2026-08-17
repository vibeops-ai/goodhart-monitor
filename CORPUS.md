# Corpus

The pilot runs on the PhysioNet/Computing in Cardiology Challenge 2019 sepsis
dataset, version 1.0.0.

    https://physionet.org/content/challenge-2019/1.0.0/

Open Access under the ODC Attribution License. 40,336 ICU stays across two
hospital systems, one pipe-separated file per patient, one row per hour. The
`SepsisLabel` column is positioned six hours ahead of clinical onset by the
challenge organisers, which is what makes "warns early" separable from "notices
afterwards".

`pilots/physionet_sepsis/fetch.sh` pulls it. The index files
`data/index_setA.txt` and `data/index_setB.txt` pin the exact file list.

## What is real and what is derived

| | |
|---|---|
| vitals, labs, demographics | recorded in the corpus, unmodified |
| `SepsisLabel` | the corpus's adjudicated label, unmodified |
| the model being verified | trained here by `train_maker.py` on hospital A only |
| the scores | that model's output on hospital B, which it never saw |
| calendar timestamps | derived. The corpus carries ICU hour and no clock. Hours are projected onto a fixed epoch so timestamps are reproducible; every event carries `icu_hour` and `calendar_is_derived` |
| the organisation and deployment names | placeholders. No health system is named, and there is no live deployment |

There is no vendor and no customer in this pilot. It exists to show what the
checker does to a model card whose numbers are all true.

## Reproducing

    ./pilots/physionet_sepsis/run.sh                            # the record
    python pilots/physionet_sepsis/emit_contracts.py            # contract objects

`run.sh` is deterministic. The record's `record_sha256` is the SHA-256 of the
canonical JSON of every other field, so a rerun that produces different bytes
means something changed.
