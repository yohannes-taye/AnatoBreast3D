"""Run MammaSegNet inference on a video and overlay breast/nipple tracking."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.infer_demo import preprocess, postprocess_mask, remap_boxes, visualize
from train_breastseg import build_model, load_config
from utils.utils import cvtColor


def format_ratio(value: float) -> str:
    return f"{value * 100:.1f}%"


def compute_mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cx = int(xs.mean())
    cy = int(ys.mean())
    return cx, cy


def compute_box_center(box: np.ndarray) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    cx = int(0.5 * (x1 + x2))
    cy = int(0.5 * (y1 + y2))
    return cx, cy


def draw_trail(frame: np.ndarray, points: Deque[Tuple[int, int]], color: Tuple[int, int, int], label: str) -> None:
    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    if points:
        cv2.circle(frame, points[-1], 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (points[-1][0] + 6, points[-1][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video breast/nipple tracking demo.")
    parser.add_argument("--config", default="configs/breastseg/breastseg_b2_inplace.py", help="Config file.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint/breastseg.pth",
        help="Checkpoint path with trained weights.",
    )
    parser.add_argument("--video", default="tests/three_images_video.mp4", help="Input video path.")
    parser.add_argument(
        "--output",
        default="tests/output_vis/three_images_video_tracked.mp4",
        help="Path to save the visualized video.",
    )
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold.")
    parser.add_argument("--topk", type=int, default=2, help="Top-k detections to keep per frame.")
    parser.add_argument("--trail", type=int, default=30, help="Number of past centers to show for trajectories.")
    return parser.parse_args()


def prepare_model(cfg_path: str, checkpoint_path: str) -> Tuple[torch.nn.Module, torch.device, tuple[int, int], dict]:
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, init_from=None)
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    input_size = tuple(cfg["data"].get("input_size", (512, 512)))
    test_cfg = cfg.get("test", {})
    return model, device, input_size, test_cfg


def process_video(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple[int, int],
    test_cfg: dict,
) -> Path:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video frame size.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create writer for {output_path}")

    breast_trail: Deque[Tuple[int, int]] = deque(maxlen=max(1, args.trail))
    nipple_trail: Deque[Tuple[int, int]] = deque(maxlen=max(1, args.trail))
    frame_logs: List[Tuple[int, float, float]] = []
    inference_times: List[float] = []

    frame_idx = 0
    nms_iou = test_cfg.get("nms_iou", 0.5)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cvtColor(Image.fromarray(rgb))
        tensor, window, orig_size = preprocess(image, input_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            tic = time.time()
            outputs = model(
                tensor,
                mode="predict",
                score_thr=args.score_thr,
                topk=args.topk,
                nms_iou=nms_iou,
            )
            inference_times.append(time.time() - tic)

        result = outputs[0]
        mask = postprocess_mask(result["seg_mask"], orig_size, input_size, window)
        boxes = remap_boxes(result["detection"].boxes, orig_size, input_size, window)
        scores = result["detection"].scores.detach().cpu()
        overlay = visualize(np.array(image, dtype=np.uint8), mask, boxes, scores)

        breast_center = compute_mask_centroid(mask)
        if breast_center:
            breast_trail.append(breast_center)
            draw_trail(overlay, breast_trail, (255, 255, 0), "breast")

        nipple_center = None
        if boxes.shape[0] > 0:
            nipple_center = compute_box_center(boxes[0])
            nipple_trail.append(nipple_center)
            draw_trail(overlay, nipple_trail, (255, 0, 255), "nipple")

        coverage = float(mask.mean())
        best_score = float(scores.max().item()) if scores.numel() else 0.0
        frame_logs.append((frame_idx, coverage, best_score))

        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    fps_proc = frame_idx / max(elapsed, 1e-6)
    avg_inf = np.mean(inference_times) if inference_times else 0.0

    print("\n[Video Summary]")
    print(f"Frames processed : {frame_idx}")
    print(f"Wall-clock FPS   : {fps_proc:.2f}")
    print(f"Avg infer time   : {avg_inf*1000:.1f} ms/frame")
    print(f"Output saved to  : {output_path}")

    if frame_logs:
        coverages = [c for _, c, _ in frame_logs]
        best_scores = [s for _, _, s in frame_logs]
        print(
            f"Mask coverage    : min={format_ratio(min(coverages))} | "
            f"max={format_ratio(max(coverages))} | mean={format_ratio(float(np.mean(coverages)))}"
        )
        print(
            f"Best nipple score: min={min(best_scores):.3f} | "
            f"max={max(best_scores):.3f} | mean={float(np.mean(best_scores)):.3f}"
        )

    return output_path


def main() -> None:
    args = parse_args()
    model, device, input_size, test_cfg = prepare_model(args.config, args.checkpoint)
    process_video(args, model, device, input_size, test_cfg)


if __name__ == "__main__":
    main()
