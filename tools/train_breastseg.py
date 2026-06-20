"""Training entry for MammaSegNet multi-task pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocess.breast_coco import BreastCocoDataset, breast_collate_fn  # noqa: E402
from nets.segmentors.breastseg_multitask import MammaSegNet  # noqa: E402
from utils.utils import seed_everything  # noqa: E402


def setup_logger(work_dir: str, log_filename: str = "train.log") -> logging.Logger:
    """Setup logger to write to both console and file.

    Args:
        work_dir: Directory to save log file
        log_filename: Name of the log file

    Returns:
        Configured logger instance
    """
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, log_filename)

    # Create logger
    logger = logging.getLogger("MammaSegNet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers

    # File handler - save all logs to file
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - display logs on terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt='%(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(path: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("breastseg_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if hasattr(module, "CONFIG"):
        return module.CONFIG
    raise AttributeError("Config file must define CONFIG dictionary.")


def build_dataloaders(cfg: Dict[str, Any], data_root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]

    # Merge multiple repeat configurations if present
    repeat_configs = []
    if data_cfg.get("repeat_samples"):
        repeat_configs.append(data_cfg["repeat_samples"])
    if data_cfg.get("dataset3_repeat"):
        repeat_configs.append(data_cfg["dataset3_repeat"])

    train_dataset = BreastCocoDataset(
        data_root=data_root,
        split=data_cfg.get("train_split", "train"),
        input_size=tuple(data_cfg.get("input_size", (512, 512))),
        multi_scale=tuple(data_cfg.get("multi_scale", (384, 640))),
        train=True,
        repeat_samples=repeat_configs if repeat_configs else None,
    )
    val_dataset = BreastCocoDataset(
        data_root=data_root,
        split=data_cfg.get("val_split", "val"),
        input_size=tuple(data_cfg.get("input_size", (512, 512))),
        multi_scale=tuple(data_cfg.get("multi_scale", (384, 640))),
        train=False,
        repeat_samples=None,
    )
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=breast_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=breast_collate_fn,
    )
    return train_loader, val_loader


def build_model(cfg: Dict[str, Any], device: torch.device, init_from: str | None) -> MammaSegNet:
    model_cfg = cfg["model"].copy()
    if init_from:
        model_cfg["init_from"] = init_from
    model = MammaSegNet(
        num_classes=model_cfg.get("num_classes", 2),
        phi=model_cfg.get("phi", "b2"),
        init_from=model_cfg.get("init_from"),
        use_prior=model_cfg.get("use_prior", True),
        loss_weights=model_cfg.get("loss_weights"),
        sigma_ratio=model_cfg.get("sigma_ratio", 0.02),
        det_head_cfg=model_cfg.get("det_head"),
    )
    return model.to(device)


def get_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    solver = cfg["solver"]
    lr = solver.get("lr", 6e-4)
    weight_decay = solver.get("weight_decay", 0.01)
    betas = tuple(solver.get("betas", (0.9, 0.999)))
    if solver.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("Only AdamW optimizer is supported.")
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def lr_lambda(epoch: int, cfg: Dict[str, Any]) -> float:
    solver = cfg["solver"]
    total_epochs = solver.get("epochs", 80)
    warmup = solver.get("warmup_epochs", 5)
    min_scale = float(solver.get("lr_min_scale", 0.0))
    min_scale = max(0.0, min(min_scale, 1.0))
    if epoch < warmup:
        return float(epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    progress = max(0.0, min(progress, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine


def train_one_epoch(
    model: MammaSegNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    log_interval: int,
    grad_clip: float,
    accum_steps: int,
    logger: logging.Logger,
) -> float:
    model.train()
    total_loss = 0.0
    accum_steps = max(1, int(accum_steps))
    use_clip = grad_clip is not None and grad_clip > 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        gt_seg = batch["gt_seg"].to(device, non_blocking=True)
        gt_boxes = [boxes.to(device) for boxes in batch["gt_boxes"]]
        seg_valid = batch["seg_valid"].to(device)
        det_valid = batch["det_valid"].to(device)
        if amp_enabled and scaler is not None:
            with autocast():
                losses, _ = model(
                    images,
                    gt_seg=gt_seg,
                    gt_boxes=gt_boxes,
                    seg_valid=seg_valid,
                    det_valid=det_valid,
                    mode="loss",
                )
                loss_total = losses["loss_total"]
            loss = loss_total / accum_steps
            scaler.scale(loss).backward()
        else:
            losses, _ = model(
                images,
                gt_seg=gt_seg,
                gt_boxes=gt_boxes,
                seg_valid=seg_valid,
                det_valid=det_valid,
                mode="loss",
            )
            loss_total = losses["loss_total"]
            (loss_total / accum_steps).backward()

        total_loss += loss_total.item()
        should_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(loader))
        if should_step:
            if amp_enabled and scaler is not None:
                scaler.unscale_(optimizer)
                if use_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % max(1, log_interval) == 0:
            # 详细的损失信息打印
            loss_seg = losses.get("loss_seg", torch.tensor(0.0)).item()
            loss_det = losses.get("loss_det", torch.tensor(0.0)).item()
            loss_boundary = losses.get("loss_boundary", torch.tensor(0.0)).item()
            logger.info(
                f"[Train] Step [{step + 1}/{len(loader)}] "
                f"Total Loss: {loss_total.item():.4f} | "
                f"Seg: {loss_seg:.4f} | "
                f"Det: {loss_det:.4f} | "
                f"Boundary: {loss_boundary:.4f}"
            )
    return total_loss / max(len(loader), 1)


def evaluate(
    model: MammaSegNet,
    loader: DataLoader,
    device: torch.device,
    eval_cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    dice_scores: List[float] = []
    iou_scores: List[float] = []
    det_scores: List[float] = []
    det_small_scores: List[float] = []
    total_images = 0
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            outputs = model(
                images,
                mode="predict",
                score_thr=eval_cfg.get("score_thr"),
                topk=eval_cfg.get("topk"),
                nms_iou=eval_cfg.get("nms_iou"),
            )
            seg_gt = batch["gt_seg"].to(device)
            seg_valid = batch["seg_valid"].to(device)
            gt_boxes = batch["gt_boxes"]
            det_valid = batch["det_valid"]

            for idx, result in enumerate(outputs):
                total_images += 1
                if seg_valid[idx]:
                    dice, iou = segmentation_metrics(result["seg_mask"].to(device), seg_gt[idx : idx + 1])
                    dice_scores.append(dice)
                    iou_scores.append(iou)
                if det_valid[idx]:
                    ap, ap_small = detection_metrics(
                        result["detection"].boxes.cpu(),
                        result["detection"].scores.cpu(),
                        gt_boxes[idx].cpu(),
                        iou_thr=0.5,
                    )
                    det_scores.append(ap)
                    det_small_scores.append(ap_small)
    elapsed = time.time() - start
    fps = total_images / max(elapsed, 1e-6)
    metrics = {
        "dice": float(torch.tensor(dice_scores).mean().item()) if dice_scores else 0.0,
        "miou": float(torch.tensor(iou_scores).mean().item()) if iou_scores else 0.0,
        "ap": float(torch.tensor(det_scores).mean().item()) if det_scores else 0.0,
        "ap_small": float(torch.tensor(det_small_scores).mean().item()) if det_small_scores else 0.0,
        "fps": fps,
    }
    return metrics


def segmentation_metrics(pred_mask: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    pred = pred_mask.view(-1)
    target = gt.view(-1)
    valid = target != 255
    pred = pred[valid]
    target = target[valid]
    pred_pos = (pred == 1).float()
    target_pos = (target == 1).float()
    intersection = (pred_pos * target_pos).sum()
    dice = (2 * intersection + 1e-6) / (pred_pos.sum() + target_pos.sum() + 1e-6)
    union = pred_pos.sum() + target_pos.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice.item(), iou.item()


def detection_metrics(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thr: float = 0.5,
) -> tuple[float, float]:
    if gt_boxes.numel() == 0 and boxes.numel() == 0:
        return 0.0, 0.0
    if gt_boxes.numel() == 0:
        return 0.0, 0.0
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]
    matched = torch.zeros_like(scores, dtype=torch.float32)
    small_matched = torch.zeros_like(scores, dtype=torch.float32)
    gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    gt_small = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])) <= (32 ** 2)

    for i, box in enumerate(boxes):
        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        max_iou, idx = torch.max(ious, dim=0)
        if max_iou >= iou_thr and not gt_used[idx]:
            matched[i] = 1.0
            gt_used[idx] = True
            if gt_small[idx]:
                small_matched[i] = 1.0

    ap = average_precision(matched, scores, gt_boxes.shape[0])
    small_total = int(gt_small.sum().item())
    ap_small = average_precision(small_matched, scores, max(small_total, 1))
    return ap, ap_small


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def average_precision(matches: torch.Tensor, scores: torch.Tensor, num_gt: int) -> float:
    if matches.numel() == 0 or num_gt == 0:
        return 0.0
    order = scores.argsort(descending=True)
    matches = matches[order]
    precision = torch.cumsum(matches, dim=0) / torch.arange(1, matches.numel() + 1, device=matches.device)
    recall = torch.cumsum(matches, dim=0) / max(num_gt, 1)
    precision = torch.cat([torch.tensor([0.0]), precision.cpu(), torch.tensor([0.0])])
    recall = torch.cat([torch.tensor([0.0]), recall.cpu(), torch.tensor([1.0])])
    for i in range(precision.numel() - 1, 0, -1):
        precision[i - 1] = torch.max(precision[i - 1], precision[i])
    indices = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap.item()


def save_checkpoint(model: nn.Module, path: str, logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    logger.info(f"[Checkpoint] Saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MammaSegNet multi-task model.")
    parser.add_argument("--config", required=True, help="Path to config python file.")
    parser.add_argument("--data-root", required=False, help="Dataset root directory.")
    parser.add_argument("--init-from", required=False, help="Override initialization checkpoint.")
    parser.add_argument("--work-dir", required=False, help="Work directory for checkpoints.")
    parser.add_argument(
        "--deploy-to",
        required=False,
        help="If set, copy the best checkpoint here after training "
        "(any existing file is backed up to <path>.bak first).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("experiment", {}).get("seed", 11))

    data_root = args.data_root or cfg["data"]["root"]
    work_dir = args.work_dir or cfg["experiment"]["output_dir"]
    os.makedirs(work_dir, exist_ok=True)

    # Setup logger and TensorBoard.
    # Keep generated training logs separate from SDK/runtime logs.
    project_root = Path(__file__).resolve().parents[1]
    log_dir = os.path.join(project_root, "logs", "training")
    os.makedirs(log_dir, exist_ok=True)

    # Generate unique log filename based on work_dir name and timestamp
    exp_name = os.path.basename(work_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{exp_name}_{timestamp}.log"

    logger = setup_logger(log_dir, log_filename)
    tb_writer = SummaryWriter(log_dir=os.path.join(work_dir, "tensorboard"))
    logger.info(f"Logging to: {os.path.join(log_dir, log_filename)}")
    logger.info(f"TensorBoard logs: {os.path.join(work_dir, 'tensorboard')}")

    batch_size = cfg["solver"].get("batch_size", 8)
    train_loader, val_loader = build_dataloaders(cfg, data_root, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, args.init_from)
    optimizer = get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, cfg))
    amp_enabled = cfg["solver"].get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_score = -float("inf")
    best_path = os.path.join(work_dir, "best.pth")
    alpha = cfg["evaluation"].get("dice_ap_alpha", 0.2)
    log_interval = cfg["experiment"].get("log_interval", 10)

    logger.info("\n" + "="*80)
    logger.info(f"Starting Training - Total Epochs: {cfg['solver']['epochs']}")
    logger.info(f"Train Samples: {len(train_loader.dataset)} | Val Samples: {len(val_loader.dataset)}")
    logger.info(f"Device: {device} | Batch Size: {batch_size} | AMP: {amp_enabled}")
    logger.info(f"Initial LR: {cfg['solver'].get('lr', 6e-4)}")
    logger.info("="*80 + "\n")

    for epoch in range(cfg["solver"].get("epochs", 80)):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch [{epoch + 1}/{cfg['solver']['epochs']}] | LR: {current_lr:.6f}")
        logger.info(f"{'='*80}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            log_interval,
            cfg["solver"].get("grad_clip", 1.0),
            cfg["solver"].get("accum_steps", 1),
            logger,
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"\n[Train Summary] Epoch {epoch + 1} | Avg Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")

        metrics = evaluate(model, val_loader, device, cfg["test"])
        metric_score = metrics["dice"] + alpha * metrics["ap_small"]
        logger.info(
            f"[Validation] Dice: {metrics['dice']:.4f} | mIoU: {metrics['miou']:.4f} | "
            f"AP: {metrics['ap']:.4f} | APs: {metrics['ap_small']:.4f} | FPS: {metrics['fps']:.2f}"
        )
        logger.info(f"[Metric Score] Combined Score (Dice + {alpha}*APs): {metric_score:.4f}")

        # Log to TensorBoard
        tb_writer.add_scalar("Train/Loss", train_loss, epoch + 1)
        tb_writer.add_scalar("Train/LR", current_lr, epoch + 1)
        tb_writer.add_scalar("Val/Dice", metrics["dice"], epoch + 1)
        tb_writer.add_scalar("Val/mIoU", metrics["miou"], epoch + 1)
        tb_writer.add_scalar("Val/AP", metrics["ap"], epoch + 1)
        tb_writer.add_scalar("Val/AP_small", metrics["ap_small"], epoch + 1)
        tb_writer.add_scalar("Val/MetricScore", metric_score, epoch + 1)
        tb_writer.add_scalar("Val/FPS", metrics["fps"], epoch + 1)

        save_checkpoint(model, os.path.join(work_dir, f"epoch_{epoch + 1:03d}.pth"), logger)
        if metric_score > best_score:
            improvement = metric_score - best_score
            best_score = metric_score
            save_checkpoint(model, best_path, logger)
            logger.info(f"🌟 [NEW BEST] Score: {metric_score:.4f} (↑{improvement:.4f}) | Dice: {metrics['dice']:.4f} | APs: {metrics['ap_small']:.4f}")
        else:
            logger.info(f"[Best Score] {best_score:.4f} (no improvement)")
        logger.info("="*80)

    # Close TensorBoard writer
    tb_writer.close()

    logger.info("\n" + "="*80)
    logger.info("🎉 Training Completed!")
    logger.info(f"Best Model Score: {best_score:.4f}")
    logger.info(f"Best Checkpoint: {best_path}")
    logger.info(f"Total Epochs: {cfg['solver']['epochs']}")
    logger.info(f"All checkpoints saved in: {work_dir}")
    logger.info("="*80)

    if args.deploy_to:
        deploy_target = Path(args.deploy_to)
        if not os.path.isfile(best_path):
            logger.warning(f"[Deploy] Best checkpoint not found at {best_path}; skipping deploy.")
        else:
            deploy_target.parent.mkdir(parents=True, exist_ok=True)
            if deploy_target.is_file():
                backup_path = deploy_target.with_name(deploy_target.name + ".bak")
                shutil.copy2(deploy_target, backup_path)
                logger.info(f"[Deploy] Backed up existing checkpoint to {backup_path}")
            shutil.copy2(best_path, deploy_target)
            logger.info(f"[Deploy] Deployed best checkpoint to {deploy_target}")

    logger.info("\n✅ DONE: MammaSegNet (B2) training pipeline ready for inference.")


if __name__ == "__main__":
    main()
