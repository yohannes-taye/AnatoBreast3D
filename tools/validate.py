"""Validation script for MammaSegNet checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loaders.breast_coco import BreastCocoDataset, breast_collate_fn
from train_breastseg import build_model, evaluate, load_config


def build_val_loader(cfg, data_root):
    dataset = BreastCocoDataset(
        data_root=data_root,
        split=cfg["data"].get("val_split", "val"),
        input_size=tuple(cfg["data"].get("input_size", (512, 512))),
        multi_scale=tuple(cfg["data"].get("input_size", (512, 512))),
        train=False,
    )
    batch_size = cfg["solver"].get("batch_size", 8)
    num_workers = cfg["data"].get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=breast_collate_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MammaSegNet checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--data-root", required=False, help="Dataset root override.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data"]["root"]
    loader = build_val_loader(cfg, data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, init_from=None)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    metrics = evaluate(model, loader, device, cfg["test"])
    print(
        f"[Validate] Dice={metrics['dice']:.4f} mIoU={metrics['miou']:.4f} "
        f"AP={metrics['ap']:.4f} APs={metrics['ap_small']:.4f} FPS={metrics['fps']:.2f}"
    )


if __name__ == "__main__":
    main()
