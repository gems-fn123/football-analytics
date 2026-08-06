#!/usr/bin/env bash
# Full-length runs on all three clips. Defaults untouched: no threshold tuning,
# so the numbers are an honest out-of-the-box baseline.
set -uo pipefail
cd "$(dirname "$0")/.."

run() {
  local clip="$1" video="$2"
  echo "=============================== $clip ==============================="
  .venv/Scripts/footy.exe run \
    --video "$video" \
    --config configs/pipeline.yaml \
    --match "configs/match/${clip}.yaml"
  echo "exit=$? for $clip"
}

run clip0 "data/raw/smoke_clip.mp4"
run clip1 "data/raw/smoke_clip (1).mp4"
run clip2 "data/raw/smoke_clip (2).mp4"
echo "ALL_RUNS_COMPLETE"
