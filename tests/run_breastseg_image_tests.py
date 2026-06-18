"""Run BreastSegFormer inference on test images and report detection/segmentation stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.infer_demo import list_images, preprocess, postprocess_mask, remap_boxes, visualize
from train_breastseg import build_model, load_config
from utils.utils import cvtColor


def format_ratio(value: float) -> str:
    return f"{value * 100:.1f}%"


def run_inference(
    image_paths: List[Path],
    cfg_path: str,
    checkpoint_path: str,
    save_dir: Path,
    score_thr: float,
    topk: int,
) -> None:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, init_from=None)
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(cfg["data"].get("input_size", (512, 512)))
    stats = []
    for path in image_paths:
        image = cvtColor(Image.open(path))
        tensor, window, orig_size = preprocess(image, input_size)
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = model(
                tensor,
                mode="predict",
                score_thr=score_thr,
                topk=topk,
                nms_iou=cfg["test"].get("nms_iou", 0.5),
            )
        result = outputs[0]
        mask = postprocess_mask(result["seg_mask"], orig_size, input_size, window)
        boxes = remap_boxes(result["detection"].boxes, orig_size, input_size, window)
        overlay = visualize(np.array(image, dtype=np.uint8), mask, boxes, result["detection"].scores.cpu())
        out_path = save_dir / f"{path.stem}_vis.png"
        Image.fromarray(overlay).save(out_path)

        coverage = float(mask.mean())
        num_boxes = int(boxes.shape[0])
        best_score = float(result["detection"].scores.max().item()) if result["detection"].scores.numel() else 0.0
        stats.append((path.name, num_boxes, best_score, coverage, out_path))

    print("\n[Summary]")
    for name, num_boxes, best_score, coverage, out_path in stats:
        print(
            f"{name}: boxes={num_boxes} | best_score={best_score:.3f} | mask_cover={format_ratio(coverage)} | vis={out_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test BreastSegFormer on sample images.")
    parser.add_argument("--config", default="configs/breastseg/breastseg_b2_inplace.py", help="Config file.")
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/breastseg_b2/epoch_043.pth",
        help="Checkpoint path (usually best.pth).",
    )
    parser.add_argument("--input", default="tests", help="Image file or directory.")
    parser.add_argument("--save-dir", default="tests/output_vis", help="Directory for visualization outputs.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold.")
    parser.add_argument("--topk", type=int, default=2, help="Top-k detections to keep.")
    args = parser.parse_args()

    paths = [p for p in list_images(args.input) if not p.stem.lower().endswith("_vis")]
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input}")
    run_inference(paths, args.config, args.checkpoint, Path(args.save_dir), args.score_thr, args.topk)


if __name__ == "__main__":
    main()
