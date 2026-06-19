"""Live 3-camera loop: streams the fused 3D reconstruction plus per-camera RGB,
depth, segmentation masks and nipple boxes to an embedded rerun viewer, which the
browser page (viewer/live.html) shows inside an iframe.

Keeps the 3 Femto Bolt cameras open (Sync Hub Pro: cam0 primary started last). Each
iteration logs everything to rerun (rr.serve_web) and still writes the minimal files
the active-learning annotator depends on:
  live/cam{N}_rgb.jpg, live/cam{N}_mask.png   (frozen frame + predicted mask)
  fused_breast.ply                            (cleaned, optional Poisson surface)
  live_manifest.json                          (points, fps, perception, rerun_url)
Pass --no-rerun to capture/annotate without the live viewer stream.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, cv2, open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recon3d.cli import load_recon_config
from recon3d.calibration.io import load_extrinsics
from recon3d.sensors.femto_bolt import FemtoBoltCamera
from recon3d.sensors.rig import MultiFemtoBoltRig
from recon3d.perception.inference import BreastPerceptionAdapter
from recon3d.config import CameraIntrinsics, DepthConfig, FusionConfig, CameraExtrinsics
from recon3d.geometry.backproject import backproject_masked_depth
from recon3d.geometry.fusion import PointCloudFusionBackend

SPECS = [("cam1", "CL8T7540197", "secondary"),
         ("cam2", "CL8T75400XA", "secondary"),
         ("cam0", "CL8T754004D", "primary")]


def flying(z, valid, t):
    o = np.zeros_like(z, bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        zs = np.roll(np.roll(z, dy, 0), dx, 1); vs = np.roll(np.roll(valid, dy, 0), dx, 1)
        o |= (valid & vs) & (np.abs(z - zs) > t)
    return o


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def build_blueprint(rrb, cam_ids):
    """Build the embedded rerun layout: fused 3D on top, per-camera RGB+depth below.

    Input shape/unit/frame: ``rrb`` is the ``rerun.blueprint`` module; ``cam_ids`` are
    the camera entity-path stems (e.g. ``cam0``).
    Return type: ``rerun.blueprint.Blueprint`` with side panels collapsed for embedding.
    """
    rgb_views = [rrb.Spatial2DView(origin=f"cams/{c}", name=f"{c} rgb") for c in cam_ids]
    depth_views = [rrb.Spatial2DView(origin=f"depth/{c}", name=f"{c} depth") for c in cam_ids]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="world", name="Fused 3D"),
            rrb.Grid(contents=rgb_views + depth_views, grid_columns=len(cam_ids)),
            row_shares=[2.0, 1.4],
        ),
        collapse_panels=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/recon3d_live")
    ap.add_argument("--config", default="runs/_tmp_live.yaml")
    ap.add_argument("--erode", type=int, default=4)
    ap.add_argument("--grad", type=float, default=0.015)
    ap.add_argument("--voxel", type=float, default=0.003)
    ap.add_argument("--outlier-std", type=float, default=2.0)
    ap.add_argument("--depth-min", type=float, default=0.25)
    ap.add_argument("--depth-max", type=float, default=1.2)
    ap.add_argument("--rgb-width", type=int, default=640)
    ap.add_argument("--mesh", action="store_true", help="Poisson surface (slower, smoother).")
    ap.add_argument("--color-size", nargs=2, type=int, default=[1280, 720])
    ap.add_argument(
        "--reuse-existing-sync-config",
        action="store_true",
        help="Do not rewrite camera sync roles; use roles already stored on the devices.",
    )
    ap.add_argument("--no-rerun", action="store_true", help="Disable the embedded rerun viewer stream.")
    ap.add_argument("--rerun-web-port", type=int, default=9090, help="Port hosting the rerun web viewer assets.")
    ap.add_argument("--rerun-grpc-port", type=int, default=9876, help="Port hosting the rerun gRPC data stream.")
    ap.add_argument("--rerun-mem", default="25%", help="rerun server memory budget; old frames are dropped past it.")
    ap.add_argument(
        "--sync-diagnostics",
        action="store_true",
        help="Write SDK FIFO/timestamp diagnostics to JSONL without changing capture policy.",
    )
    ap.add_argument(
        "--sync-diagnostics-path",
        default=None,
        help="JSONL path for --sync-diagnostics; defaults to <out>/sync_diagnostics.jsonl.",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="Also save the rerun stream to runs/.../recordings/*.rrd for later replay.",
    )
    args = ap.parse_args()

    out = Path(args.out); live = out / "live"; live.mkdir(parents=True, exist_ok=True)
    sync_diag_path = (
        Path(args.sync_diagnostics_path)
        if args.sync_diagnostics and args.sync_diagnostics_path
        else out / "sync_diagnostics.jsonl"
        if args.sync_diagnostics
        else None
    )
    cfg, ext_path = load_recon_config(Path(args.config), out)
    ext = load_extrinsics(ext_path)
    cams = [
        FemtoBoltCamera(
            c,
            serial_number=s,
            sync_mode=m,
            align_mode="sw",
            color_size=tuple(args.color_size),
            apply_sync_config=not args.reuse_existing_sync_config,
        )
        for c, s, m in SPECS
    ]
    rig = MultiFemtoBoltRig(
        cams,
        max_timestamp_drift_ms=cfg.sync_max_drift_ms,
        sync_diagnostics_path=sync_diag_path,
    )
    perc = BreastPerceptionAdapter(config=cfg)
    fusion = PointCloudFusionBackend(FusionConfig(voxel_size_m=args.voxel, outlier_std_ratio=args.outlier_std))
    cam_centers = [ext[c].T_world_cam[:3, 3] for c, _, _ in SPECS if c in ext]

    RR = None
    REC = None
    RECS: list = []  # rerun recordings to tee each log to (default sink = live; +file when --record)
    rerun_url = None
    rerun_error = None
    if not args.no_rerun:
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            cam_order = sorted({c for c, _, _ in SPECS})  # cam0, cam1, cam2
            blueprint = build_blueprint(rrb, cam_order)
            rr.init("breast_live")
            rr.serve_web(
                open_browser=False,
                web_port=args.rerun_web_port,
                grpc_port=args.rerun_grpc_port,
                default_blueprint=blueprint,
                server_memory_limit=args.rerun_mem,
            )
            RR = rr
            RECS = [None]  # None => default recording, streamed to the live web viewer
            rerun_url = (
                f"http://localhost:{args.rerun_web_port}"
                f"?url=rerun%2Bhttp://localhost:{args.rerun_grpc_port}/proxy"
            )
            print(f"rerun viewer streaming at {rerun_url}", flush=True)
            if args.record:
                rec_dir = out / "recordings"
                rec_dir.mkdir(parents=True, exist_ok=True)
                rec_path = rec_dir / time.strftime("recon_%Y%m%d_%H%M%S.rrd")
                REC = rr.new_recording("breast_live")
                rr.save(str(rec_path), default_blueprint=blueprint, recording=REC)
                RECS.append(REC)
                print(f"recording session to {rec_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - viewer is optional; keep capturing without it
            rerun_error = f"{type(exc).__name__}: {exc}"
            print(f"rerun disabled ({rerun_error})", flush=True)
            RR = None
            RECS = []

    def rr_set_clock(seq: int, wall: float) -> None:
        for rec in RECS:
            RR.set_time("frame", sequence=seq, recording=rec)
            RR.set_time("time", timestamp=wall, recording=rec)

    def rr_emit(path: str, archetype) -> None:
        for rec in RECS:
            RR.log(path, archetype, recording=rec)

    def read_live_frames() -> dict:
        return rig.read_synchronized_frames()

    rig.start()
    print("Live recon started. Ctrl+C to stop.", flush=True)
    try:
        for _ in range(15):
            try:
                rig.read_synchronized_frames()
            except Exception:
                pass
        frame_i = 0
        while True:
            t0 = time.perf_counter()
            frame_wall = time.time()
            try:
                frames = read_live_frames()
            except RuntimeError as exc:
                print(f"Skipping frame set: {exc}", flush=True)
                continue
            t_read = time.perf_counter()
            if RR is not None:
                rr_set_clock(frame_i, frame_wall)

            # stream RGB to disk (kept for the active-learning annotator); mirror RGB+depth to rerun
            for cid, f in frames.items():
                scale = f.depth_config.depth_scale_m
                bgr = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
                w = args.rgb_width; h = int(bgr.shape[0] * w / bgr.shape[1])
                cv2.imwrite(str(live / f"{cid}_rgb.jpg"), cv2.resize(bgr, (w, h)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if RR is not None:
                    rr_emit(f"cams/{cid}/image", RR.Image(f.rgb))
                    rr_emit(f"depth/{cid}", RR.DepthImage(
                        f.depth_mm, meter=1.0 / scale, colormap="turbo",
                        depth_range=[args.depth_min, args.depth_max],
                    ))
            t_stream = time.perf_counter()

            # perception (model forward only — timed separately from overlay/IO below)
            results = perc.predict_batch(frames)
            t_infer = time.perf_counter()
            perception = {}
            display = {}
            for cid, f in frames.items():
                r = results[cid]
                w = args.rgb_width; h = int(f.rgb.shape[0] * w / f.rgb.shape[1])
                # mask.png is the annotator's starting prediction; keep writing it
                mask_png = cv2.resize(
                    (r.breast_mask.astype(np.uint8) * 255),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imwrite(str(live / f"{cid}_mask.png"), mask_png)
                if RR is not None:
                    rr_emit(f"cams/{cid}/seg", RR.SegmentationImage(r.breast_mask.astype(np.uint8), opacity=0.5))
                    boxes = np.asarray(r.nipple_boxes_xyxy, dtype=np.float32)
                    if len(boxes):
                        labels = [f"nipple {float(s):.2f}" for s in r.nipple_scores]
                        rr_emit(f"cams/{cid}/nipple", RR.Boxes2D(
                            mins=boxes[:, :2], sizes=boxes[:, 2:] - boxes[:, :2],
                            colors=[(255, 64, 64)] * len(boxes), labels=labels,
                        ))
                    else:
                        rr_emit(f"cams/{cid}/nipple", RR.Boxes2D(mins=np.zeros((0, 2)), sizes=np.zeros((0, 2))))
                display[cid] = {
                    "width": int(w),
                    "height": int(h),
                    "source_width": int(f.rgb.shape[1]),
                    "source_height": int(f.rgb.shape[0]),
                }
                perception[cid] = {
                    "mask_pixels": int(r.breast_mask.sum()),
                    "mask_bbox_xyxy": mask_bbox(r.breast_mask),
                    "nipple_boxes_xyxy": np.asarray(r.nipple_boxes_xyxy, dtype=float).tolist(),
                    "nipple_scores": np.asarray(r.nipple_scores, dtype=float).tolist(),
                }
            t_perception = time.perf_counter()

            clouds = {}
            for cid, f in frames.items():
                mask = results[cid].breast_mask.astype(bool)
                depth = f.depth_mm; valid = depth > 0
                z = depth.astype(np.float64) * f.depth_config.depth_scale_m
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode * 2 + 1, args.erode * 2 + 1))
                m = cv2.erode(mask.astype(np.uint8), k).astype(bool) & ~flying(z, valid, args.grad) & valid
                K = f.intrinsics
                dcfg = DepthConfig(depth_scale_m=f.depth_config.depth_scale_m,
                                   min_depth_m=args.depth_min, max_depth_m=args.depth_max)
                clouds[cid] = backproject_masked_depth(depth, m, f.rgb, K, dcfg)
            fused = fusion.fuse(clouds, ext)
            t_fusion = time.perf_counter()

            if args.mesh and len(fused.points) > 500:
                fused.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
                fused.orient_normals_towards_camera_location(np.mean(cam_centers, 0))
                mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(fused, depth=8)
                dens = np.asarray(dens)
                mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.1))
                mesh = mesh.crop(fused.get_axis_aligned_bounding_box())
                if len(mesh.triangles):
                    out_cloud = mesh.sample_points_uniformly(80000)
                else:
                    out_cloud = fused
            else:
                out_cloud = fused

            o3d.io.write_point_cloud(str(out / "fused_breast.ply"), out_cloud)
            P = np.asarray(out_cloud.points)
            if RR is not None:
                cols = np.asarray(out_cloud.colors)
                rr_cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8) if len(cols) == len(P) and len(P) else None
                rr_emit("world/breast", RR.Points3D(P, colors=rr_cols, radii=0.0015))
            dt = time.perf_counter() - t0
            if RR is not None:
                rr_emit("timing/read_ms", RR.Scalars((t_read - t0) * 1000))
                rr_emit("timing/infer_ms", RR.Scalars((t_infer - t_stream) * 1000))
                rr_emit("timing/fusion_ms", RR.Scalars((t_fusion - t_perception) * 1000))
                rr_emit("timing/total_ms", RR.Scalars(dt * 1000))
            manifest = {
                "timestamp": frame_wall,
                "frame": frame_i,
                "points": len(P),
                "fps": round(1.0 / dt, 2) if dt > 0 else 0,
                "cameras": [c2 for c2, _, _ in SPECS],
                "rerun_url": rerun_url,
                "rgb": {c2: f"live/{c2}_rgb.jpg" for c2, _, _ in SPECS},
                "mask": {c2: f"live/{c2}_mask.png" for c2, _, _ in SPECS},
                "display": display,
                "perception": perception,
                "timings_ms": {
                    "read": round((t_read - t0) * 1000, 1),
                    "stream_rgb": round((t_stream - t_read) * 1000, 1),
                    "infer": round((t_infer - t_stream) * 1000, 1),
                    "perception_io": round((t_perception - t_infer) * 1000, 1),
                    "fusion": round((t_fusion - t_perception) * 1000, 1),
                    "total": round(dt * 1000, 1),
                },
            }
            if rerun_error:
                manifest["rerun_error"] = rerun_error
            (out / "live_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (out / "nipple_world.json").write_text("{}", encoding="utf-8")
            frame_i += 1
            t = manifest["timings_ms"]
            print(
                f"frame {frame_i}: {len(P)} pts, {dt*1000:.0f} ms ({manifest['fps']} fps), "
                f"read {t['read']} · infer {t['infer']} · fusion {t['fusion']} ms",
                flush=True,
            )
    finally:
        rig.stop()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
