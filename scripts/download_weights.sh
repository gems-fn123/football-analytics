#!/usr/bin/env bash
# Fetch model weights into models/weights/.
# Nothing here downloads match data. Weights only.
set -euo pipefail

WEIGHTS_DIR="${FOOTY_WEIGHTS_DIR:-models/weights}"
mkdir -p "$WEIGHTS_DIR"

echo "Target: $WEIGHTS_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Player detector - the default backend needs nothing from this script.
# ---------------------------------------------------------------------------
# configs/detector/rtdetr_coco.yaml pulls PekingU/rtdetr_r18vd from the HuggingFace
# hub on first use. Apache-2.0, no API key, cached under $HF_HOME. Warming the cache
# here is optional but keeps the first `footy run` from stalling mid-pipeline.
echo "[1/4] Player detector (default: RT-DETR, Apache-2.0, no API key)"
if python -c "import transformers" 2>/dev/null; then
  python - <<'PY'
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

model_id = "PekingU/rtdetr_r18vd"
RTDetrImageProcessor.from_pretrained(model_id)
RTDetrForObjectDetection.from_pretrained(model_id)
print(f"      cached {model_id}")
PY
else
  echo "      skipped: transformers not installed (pip install -e '.[cv]')"
fi
echo
echo "      For football roles (goalkeeper/referee as distinct classes) you need a"
echo "      fine-tune, e.g. Roboflow Universe football-players-detection. That route"
echo "      needs ROBOFLOW_API_KEY in .env. Then point configs/pipeline.yaml at"
echo "      configs/detector/rfdetr.yaml and drop the checkpoint at:"
echo "        $WEIGHTS_DIR/rfdetr_football.pt"
echo

# ---------------------------------------------------------------------------
# 2-4. Still manual.
# ---------------------------------------------------------------------------
echo "[2/4] Pitch keypoint model - broadcast and phone profiles only."
echo "      Fixed-camera users skip this entirely:"
echo "        python scripts/calibrate_fixed_camera.py --video <file>"
echo "      Check the source repo's licence before redistributing (several are GPL)."
echo
echo "[3/4] Re-ID weights - only for the botsort tracker, which is not wired up."
echo "      configs/tracker/bytetrack.yaml is the supported path and needs no weights."
echo
echo "[4/4] Ball model - optional, TrackNet-style. stages.ball currently works from"
echo "      the main detector's ball candidates and needs no extra weights; a"
echo "      dedicated heatmap model would improve coverage when one is added."
echo
echo "Done. The default configuration is runnable with what step 1 fetched."
