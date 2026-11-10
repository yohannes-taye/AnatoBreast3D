"""Exercise recording transitions without opening physical cameras."""
import json

import numpy as np
import yaml

from anatomy_guided_recon3d.calibration.io import save_extrinsics
from anatomy_guided_recon3d.config import CameraExtrinsics, CameraIntrinsics, DepthConfig, RGBDFrame
from anatomy_guided_recon3d.tracking import recording
from anatomy_guided_recon3d.tracking.runtime import write_json


def test_preview_record_save_preserves_calibration_and_releases_cameras(tmp_path, monkeypatch):
    output = tmp_path / "preview"
    output.mkdir()
    calibration = tmp_path / "calibration.json"
    save_extrinsics(calibration, {name:CameraExtrinsics(name, np.eye(4)) for name in ("cam1", "cam2")})
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"extrinsics_path":"calibration.json", "hardware":{"cameras":[
        {"id":"cam1", "serial_number":"serial1"}, {"id":"cam2", "serial_number":"serial2"}]}}))
    write_json(output / "command.json", {"id":"initial", "action":"range", "foreground_only":False})
    clock = [0.]
    def tick():
        clock[0] += .15
        return clock[0]
    monkeypatch.setattr(recording.time, "monotonic", tick)
    monkeypatch.setattr(recording, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(recording, "enable_common_clock", lambda: [])
    cameras = []
    class Camera:
        def __init__(self, cam_id, serial_number, **kwargs):
            self.cam_id, self.serial_number = cam_id, serial_number
            self.closed = False
            cameras.append(self)
        def begin_recording(self, path):
            path.write_bytes(b"test SDK stream")
        def pause_recording(self):
            pass
    class Rig:
        def __init__(self, items, **kwargs):
            self.items, self.index = items, 0
        def start(self):
            pass
        def stop(self):
            for item in self.items:
                item.closed = True
        def read_synchronized_frames(self):
            self.index += 1
            if self.index == 2:
                assert (output / "preview.json").exists()
                write_json(output / "command.json", {"id":"capture", "action":"record", "duration_s":1, "foreground_only":False})
            return {item.cam_id:RGBDFrame(item.cam_id, np.zeros((192,256,3), np.uint8),
                    np.full((192,256),600,np.uint16), self.index*100_000_000 + n*100_000,
                    CameraIntrinsics(256,192,200,200,127.5,95.5), DepthConfig(.001)) for n,item in enumerate(self.items)}
    monkeypatch.setattr(recording, "FemtoBoltCamera", Camera)
    monkeypatch.setattr(recording, "MultiFemtoBoltRig", Rig)
    recording.run(output, config)
    state = json.loads((output / "status.json").read_text())
    assert state["state"] == "recorded"
    session = tmp_path / state["session"]
    manifest = json.loads((session / "session.json").read_text())
    assert manifest["completed"]
    assert manifest["accepted_frame_sets"] > 0
    assert [c["id"] for c in manifest["cameras"]] == ["cam1", "cam2"]
    assert (session / "extrinsics.json").read_bytes() == calibration.read_bytes()
    assert all(camera.closed for camera in cameras)
