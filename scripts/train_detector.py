"""Fine-tune RF-DETR (Apache-2.0) on the Roboflow football-players-detection set.

Train-time only: the run-time pipeline treats the resulting checkpoint as a frozen
artefact (docs/architecture.md, "two clocks"). Dataset: roboflow-jvuqo/
football-players-detection-3zvbc, CC BY 4.0 - attribution recorded in NOTICE.md.
Classes (ball, goalkeeper, player, referee) match the DETECTIONS contract exactly.

The dataset is small (372 images), so this fine-tunes the smallest variant by
default. A 4 GB GPU handles rfdetr nano at batch 4 with grad accumulation; on CPU
use --smoke to verify the plumbing only.

Usage:
    python scripts/train_detector.py --dataset <roboflow>/football-players-detection-3zvbc \
        --out models/weights/rfdetr_football --epochs 50
    python scripts/train_detector.py --dataset ... --smoke   # 1 epoch, tiny budget
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="COCO export dir with train/valid/test")
    ap.add_argument("--out", default="models/weights/rfdetr_football")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="nano")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on CPU to verify plumbing")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    for split in ("train", "valid"):
        ann = dataset / split / "_annotations.coco.json"
        if not ann.exists():
            raise SystemExit(f"missing {ann}; export the dataset in COCO format")

    import rfdetr

    model = getattr(rfdetr, VARIANTS[args.variant])()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        dataset_dir=str(dataset),
        epochs=1 if args.smoke else args.epochs,
        batch_size=1 if args.smoke else args.batch_size,
        grad_accum_steps=1 if args.smoke else args.grad_accum,
        lr=1e-4,
        output_dir=str(out_dir),
    )
    print(f"done; checkpoints under {out_dir}. Point configs/detector/rfdetr.yaml at the best one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
