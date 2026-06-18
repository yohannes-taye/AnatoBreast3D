"""Short fine-tuning config for live active-learning samples."""

CONFIG = {
    "experiment": {
        "seed": 23,
        "output_dir": "./active_learning/checkpoints/default",
        "log_interval": 2,
    },
    "model": {
        "num_classes": 2,
        "phi": "b2",
        "init_from": "./work_dirs/breastseg_b2/best.pth",
        "use_prior": True,
        "sigma_ratio": 0.02,
        "loss_weights": {"seg": 1.0, "det": 0.25, "boundary": 0.1},
        "det_head": {"topk": 2, "score_thr": 0.3, "nms_iou": 0.5, "strides": (4, 8, 16)},
    },
    "data": {
        "root": "./active_learning/dataset",
        "train_split": "train",
        "val_split": "val",
        "input_size": (512, 512),
        "multi_scale": (448, 640),
        "num_workers": 0,
    },
    "solver": {
        "epochs": 12,
        "optimizer": "adamw",
        "lr": 3e-5,
        "lr_min_scale": 0.2,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "lr_schedule": "cosine",
        "warmup_epochs": 1,
        "batch_size": 2,
        "amp": True,
        "grad_clip": 1.0,
        "accum_steps": 2,
        "ema": 0.0,
    },
    "evaluation": {
        "dice_ap_alpha": 0.2,
        "period": 1,
    },
    "test": {
        "score_thr": 0.3,
        "topk": 2,
        "nms_iou": 0.5,
    },
}
