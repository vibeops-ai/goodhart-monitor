#!/usr/bin/env bash
# Reproduce GHM-0001 from nothing but the public corpus.
#
# Every step is deterministic and every step prints what it wrote, so a
# reviewer can stop after any one of them and check the intermediate artifact
# instead of taking the final record on trust.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
GHM=${GHM:-.venv/bin/goodhart-monitor}

[ -d data/setA ] && [ -d data/setB ] || ./pilots/physionet_sepsis/fetch.sh
[ -f out/B_deploy.parquet ] || $PY pilots/physionet_sepsis/build_matrices.py
[ -f out/maker.pkl ]       || $PY pilots/physionet_sepsis/train_maker.py
[ -f out/stream_B.parquet ] || $PY pilots/physionet_sepsis/to_stream.py

# From here on nothing knows the model exists. This is the hospital's half.
$GHM validate --stream out/stream_B.parquet

$GHM verify \
  --stream out/stream_B.parquet \
  --card   out/MODEL_CARD.json \
  --config pilots/physionet_sepsis/governance.toml \
  --deployment "hospital system B, 20,000 adult ICU stays, hourly scoring" \
  --limit "hospital B is a held-out system the maker never trained on; this record says nothing about hospital A" \
  --out out || true            # a FAILS exit is the expected result here

echo
echo "record:   out/record_GHM-0001.json"
echo "the read: out/record_GHM-0001.md"
