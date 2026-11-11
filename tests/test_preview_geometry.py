from types import SimpleNamespace

import numpy as np
import pytest

from anatomy_guided_recon3d.tracking.preview_geometry import add_camera_gaps, camera_frustum, make_depth_surface
from anatomy_guided_recon3d.tracking.playback import collect_samples, synchronize_samples
from anatomy_guided_recon3d.tracking.runtime import write_json
from anatomy_guided_recon3d.tracking import service


def prepared():
    depth = np.full((64, 64), .6, np.float32)
    depth[:, 32:] = .9
    depth[28:36, 28:36] = 0
    mask = np.zeros(depth.shape, bool)
    mask[20:44, 20:44] = True
    return dict(depths=depth, mask=mask, video=np.full((64,64,3), [101,160,222], np.uint8),
                intrinsics=np.array([[100,0,31.5],[0,100,31.5],[0,0,1]], np.float32), extrinsics=np.eye(4))


def test_surface_keeps_rgb_and_never_bridges_a_depth_hole_or_step():
    surface = make_depth_surface(prepared(), "scene", "fine")
    triangles = np.array(surface["triangles"]).reshape(-1, 3)
    depth = np.array(surface["depth_m"])
    assert len(triangles) > 0
    assert np.ptp(depth[triangles], axis=1).max() < .02
    assert np.all(depth > 0)
    assert np.asarray(surface["colors"]).shape == (surface["points"], 3)
    assert surface["colors"][0] == [101,160,222]
    assert max(surface["triangles"]) < surface["points"]
    assert set(surface["region"]) == {0,2,3}
    breast = make_depth_surface(prepared(), "breast", "balanced")
    assert breast["points"] < surface["points"]
    assert all(value & 1 for value in breast["region"])


def test_camera_separation_distinguishes_overlap_and_unobserved_areas():
    views = [dict(xyz=[[0,0,.6],[.01,0,.6],[2,0,.6]]), dict(xyz=[[0,0,.604],[.01,0,.604]])]
    add_camera_gaps(views)
    assert views[0]["gap_mm"] == [4,4,-1]
    assert views[1]["overlap"]["median_gap_mm"] == 4


def test_frustum_uses_intrinsics_and_the_camera_to_world_pose():
    frame=prepared()
    Twc=np.array([[0.,-1,0,.2],[1,0,0,-.1],[0,0,1,.3],[0,0,0,1]])
    frame["extrinsics"]=np.linalg.inv(Twc)
    frustum=camera_frustum(frame)
    np.testing.assert_allclose(frustum["origin_world_m"],[.2,-.1,.3])
    np.testing.assert_allclose(frustum["rays_world"][0],[.315,-.315,1])
    # Back-projecting a corner must return the same pixel through the actual K.
    ray=np.asarray(frustum["rays_world"])[2] @ Twc[:3,:3]
    pixel=frame["intrinsics"] @ ray
    np.testing.assert_allclose(pixel[:2]/pixel[2],[63,63])


def test_playback_sampling_uses_shared_time_grid_and_rejects_incompatible_clocks():
    class Camera:
        def __init__(self, start):
            self.frames=iter(SimpleNamespace(timestamp_ns=t) for t in range(start,2_000_000_000,33_333_333))
        def start(self): pass
        def stop(self): self.closed=True
        def read_aligned_frame(self): return next(self.frames)
    streams={"cam1":collect_samples(Camera(0)), "cam2":collect_samples(Camera(70_000_000))}
    pairs=synchronize_samples(streams)
    assert len(pairs) >= 8
    assert all(abs(pair["cam1"].timestamp_ns-pair["cam2"].timestamp_ns)<=50_000_000 for pair in pairs)
    with pytest.raises(ValueError,match="shared RGB-D timeline"):
        synchronize_samples({"cam1":streams["cam1"],"cam2":[SimpleNamespace(timestamp_ns=10_000_000_000+i) for i in range(3)]})


def test_view_changes_do_not_cancel_pending_recording(tmp_path, monkeypatch):
    preview=tmp_path/"runs"/"preview"
    preview.mkdir(parents=True)
    write_json(tmp_path/"runs/point_tracking/current.json",{"preview":"runs/preview"})
    write_json(preview/"status.json",{"state":"preview"})
    write_json(preview/"command.json",{"id":"capture","action":"record","duration_s":6})
    monkeypatch.setattr(service,"preview_running",lambda root:True)
    service.preview_command(tmp_path,{"action":"range","scope":"scene","detail":"fine","mask_overlay":True})
    command=service._load(preview/"command.json")
    assert command["id"] == "capture" and command["action"] == "record"
    assert command["scope"] == "scene" and command["detail"] == "fine"
    assert command["mask_overlay"]
