"""Summarize sync_diagnostics.jsonl for FIFO pool matching experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


THRESHOLDS_MS = [5.0, 10.0, 16.7, 20.0, 24.0, 30.0, 33.4, 35.0, 40.0, 50.0, 66.7, 100.0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _series_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    result: dict[str, Any] = {
        "n": len(values),
        "mean_ms": mean(values),
        "median_ms": median(values),
        "min_ms": min(values),
        "p75_ms": _percentile(values, 75),
        "p90_ms": _percentile(values, 90),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": max(values),
    }
    for threshold in THRESHOLDS_MS:
        count = sum(1 for value in values if value <= threshold)
        result[f"le_{threshold:g}ms"] = count
        result[f"le_{threshold:g}ms_pct"] = count / len(values) * 100.0
    return result


def _drift_series(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if field == "best_pool_match":
            value = _number((row.get("best_pool_match") or {}).get("best_drift_ms"))
        elif field == "chosen_pool_match":
            value = _number((row.get("chosen_pool_match") or {}).get("best_drift_ms"))
        else:
            value = _number(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _camera_depth_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_camera: dict[str, dict[str, list[float]]] = {}
    hit_max: dict[str, int] = {}
    for row in rows:
        for cam_id, camera_event in (row.get("cameras") or {}).items():
            drain = camera_event.get("drain") or {}
            bucket = by_camera.setdefault(cam_id, {"drain_count": [], "candidate_count": []})
            drain_count = _number(drain.get("drain_count"))
            if drain_count is not None:
                bucket["drain_count"].append(drain_count)
            candidate_count = _number(drain.get("candidate_count"))
            if candidate_count is None:
                candidate_count = _number((row.get("fifo_pool_depth_by_camera") or {}).get(cam_id))
            if candidate_count is not None:
                bucket["candidate_count"].append(candidate_count)
            if drain.get("hit_max_drain"):
                hit_max[cam_id] = hit_max.get(cam_id, 0) + 1
    return {
        cam_id: {
            "drain_count": _series_stats(values["drain_count"]),
            "candidate_count": _series_stats(values["candidate_count"]),
            "hit_max_drain_events": hit_max.get(cam_id, 0),
        }
        for cam_id, values in sorted(by_camera.items())
    }


def _write_plot(summary: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - summary JSON is the required artifact
        print(f"plot skipped: {exc}")
        return

    series = summary["_plot_series"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    ax = axes[0]
    for label, values, color in (
        ("old selected", series["old_flush_next_selected_drift_ms"], "#2563eb"),
        ("chosen fifo", series["chosen_pre_conversion_drift_ms"], "#f97316"),
        ("best fifo", series["best_pool_match"], "#16a34a"),
    ):
        if values:
            ax.hist(values, bins=50, alpha=0.45, label=label, color=color)
    for threshold in (5.0, 24.0, 35.0, 50.0):
        ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("cross-camera drift (ms)")
    ax.set_ylabel("event count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    for label, values, color in (
        ("old selected", series["old_flush_next_selected_drift_ms"], "#2563eb"),
        ("chosen fifo", series["chosen_pre_conversion_drift_ms"], "#f97316"),
        ("best fifo", series["best_pool_match"], "#16a34a"),
    ):
        if not values:
            continue
        ordered = sorted(values)
        y = [(index + 1) / len(ordered) * 100.0 for index in range(len(ordered))]
        ax.plot(ordered, y, label=label, color=color, linewidth=2)
    for threshold in (5.0, 24.0, 35.0, 50.0):
        ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("drift threshold (ms)")
    ax.set_ylabel("events <= threshold (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=160)


def summarize(path: Path, trim_fraction: float) -> dict[str, Any]:
    rows = _read_jsonl(path)
    trim_each = int(len(rows) * trim_fraction)
    trimmed = rows[trim_each : len(rows) - trim_each] if trim_each else rows
    plot_series = {
        "old_flush_next_selected_drift_ms": _drift_series(trimmed, "old_flush_next_selected_drift_ms"),
        "chosen_pre_conversion_drift_ms": _drift_series(trimmed, "chosen_pre_conversion_drift_ms"),
        "pre_conversion_drift_ms": _drift_series(trimmed, "pre_conversion_drift_ms"),
        "best_pool_match": _drift_series(trimmed, "best_pool_match"),
        "chosen_pool_match": _drift_series(trimmed, "chosen_pool_match"),
    }
    return {
        "input": str(path),
        "raw_events": len(rows),
        "trim_fraction": trim_fraction,
        "trim_each_side_events": trim_each,
        "trimmed_events": len(trimmed),
        "accepted": sum(1 for row in trimmed if row.get("accepted") is True),
        "dropped": sum(1 for row in trimmed if row.get("accepted") is False),
        "selected_bad_but_fifo_good": sum(1 for row in trimmed if row.get("selected_bad_but_fifo_good") is True),
        "drift_ms": {name: _series_stats(values) for name, values in plot_series.items()},
        "fifo_depth_by_camera": _camera_depth_stats(trimmed),
        "_plot_series": plot_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to sync_diagnostics.jsonl")
    parser.add_argument("--trim", type=float, default=0.10, help="Fraction to trim from each side.")
    parser.add_argument("--out-json", type=Path, default=None, help="Summary JSON output path.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG plot output path.")
    args = parser.parse_args()

    summary = summarize(args.jsonl, args.trim)
    plot_series = summary.pop("_plot_series")
    output_json = args.out_json or args.jsonl.with_name("sync_diagnostics_summary.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.plot is not None:
        summary["_plot_series"] = plot_series
        _write_plot(summary, args.plot)
        summary.pop("_plot_series", None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
