#!/usr/bin/env bash
# Fetch model weights into models/weights/.
# Nothing here downloads match data. Weights only.
set -euo pipefail

WEIGHTS_DIR="${FOOTY_WEIGHTS_DIR:-models/weights}"
mkdir -p "$WEIGHTS_DIR"

echo "Target: $WEIGHTS_DIR"
echo
echo "TODO: fill in the URLs for your chosen backends."
echo
echo "  1. Player detector"
echo "     Roboflow Universe football-players-detection, or your own fine-tune."
echo "     Requires ROBOFLOW_API_KEY in .env."
echo
echo "  2. Pitch keypoint model (broadcast and phone profiles only)"
echo "     From a calibration repo. Check its licence before redistributing."
echo
echo "  3. Re-ID weights (only if using the botsort tracker)"
echo
echo "  4. Ball model (optional, TrackNet-style)"
echo
echo "Fixed-camera users can skip item 2 entirely: run"
echo "  python scripts/calibrate_fixed_camera.py --video <file>"
