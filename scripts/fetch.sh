#!/usr/bin/env bash
# Pull the PhysioNet/CinC 2019 sepsis challenge training data (Open Access).
# 40,336 patients as one .psv per patient, two real hospital systems (A and B).
# Idempotent: skips files already present and non-empty.
set -euo pipefail
cd "$(dirname "$0")/../data"
BASE=https://physionet.org/files/challenge-2019/1.0.0/training
for S in A B; do
  mkdir -p "set$S"
  awk '{print}' "index_set$S.txt" | while read -r f; do
    [ -s "set$S/$f" ] || echo "$BASE/training_set$S/$f"
  done > "todo_$S.txt"
  n=$(wc -l < "todo_$S.txt" | tr -d ' ')
  echo "set$S: $n to fetch"
  [ "$n" -eq 0 ] && continue
  ( cd "set$S" && xargs -P 24 -n 40 curl -s --retry 3 --remote-name-all < "../todo_$S.txt" )
done
echo "done: A=$(ls setA | wc -l) B=$(ls setB | wc -l)"
