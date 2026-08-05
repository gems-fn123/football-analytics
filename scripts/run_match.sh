#!/usr/bin/env bash
# End-to-end run for one match. Smoke test first, then the full thing.
set -euo pipefail

VIDEO="${1:?usage: run_match.sh <video> [match_config]}"
MATCH="${2:-configs/match/example_match.yaml}"

echo "== probe =="
footy inspect --video "$VIDEO"

echo "== smoke test, 300 frames =="
footy run --video "$VIDEO" --config configs/pipeline.yaml --match "$MATCH" --max-frames 300

echo "Check data/processed/*/tracks.parquet before continuing."
read -r -p "Run the full match? [y/N] " reply
[[ "$reply" == "y" ]] || exit 0

echo "== full match =="
time footy run --video "$VIDEO" --config configs/pipeline.yaml --match "$MATCH"
