"""Replay a saved rerun recording (.rrd) in an embedded web viewer.

Loads a recording produced by ``live_recon.py --record`` and serves it on the
rerun web viewer so viewer/replay.html can embed it. This module deliberately
imports only rerun (no torch/open3d), so it starts fast and needs no GPU.

    python tools/replay_recon.py --file runs/recon3d_live/recordings/recon_*.rrd
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def build_blueprint(rrb, cam_ids):
    """Same fused-3D-over-2D layout as live_recon.build_blueprint, panels collapsed.

    Input shape/unit/frame: ``rrb`` is ``rerun.blueprint``; ``cam_ids`` are camera
    entity-path stems (e.g. ``cam0``).
    Return type: ``rerun.blueprint.Blueprint``.
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
    ap = argparse.ArgumentParser(description="Replay a saved rerun .rrd recording.")
    ap.add_argument("--file", required=True, help="Path to the .rrd recording to replay.")
    ap.add_argument("--web-port", type=int, default=9091, help="Port hosting the rerun web viewer.")
    ap.add_argument("--grpc-port", type=int, default=9877, help="Port hosting the rerun data stream.")
    ap.add_argument("--cameras", nargs="*", default=["cam0", "cam1", "cam2"], help="Camera ids for the layout.")
    args = ap.parse_args()

    rrd = Path(args.file)
    if not rrd.is_file():
        raise SystemExit(f"Recording not found: {rrd}")

    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("breast_replay")
    rr.serve_web(
        open_browser=False,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
        default_blueprint=build_blueprint(rrb, sorted(args.cameras)),
    )
    rr.log_file_from_path(str(rrd))
    url = f"http://localhost:{args.web_port}?url=rerun%2Bhttp://localhost:{args.grpc_port}/proxy"
    print(f"replaying {rrd.name} at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
