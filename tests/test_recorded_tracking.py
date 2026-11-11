import json
from pathlib import Path

import cv2
import numpy as np

from anatomy_guided_recon3d.config import CameraExtrinsics
from anatomy_guided_recon3d.calibration.io import save_extrinsics
from anatomy_guided_recon3d.tracking.cli import capture_recorded
from anatomy_guided_recon3d.tracking.sequence import validate_sequence
from anatomy_guided_recon3d.tracking.worker import project_world_tracks


def test_full_recording_samples_source_time_and_uses_session_calibration(tmp_path):
    session=tmp_path/'session';camera=session/'cam2'
    (camera/'rgb').mkdir(parents=True);(camera/'depth').mkdir()
    timestamps=[1_000_000_000+i*33_333_333 for i in range(76)]
    metadata=dict(intrinsics=dict(width=64,height=64,fx=80,fy=80,cx=31.5,cy=31.5),
                  depth_config=dict(depth_scale_m=.001,min_depth_m=.25,max_depth_m=1.2),timestamps_ns=timestamps)
    (camera/'metadata.json').write_text(json.dumps(metadata))
    for i in range(len(timestamps)):
        cv2.imwrite(str(camera/'rgb'/f'{i:06d}.png'),np.full((64,64,3),[100,130,170],np.uint8))
        cv2.imwrite(str(camera/'depth'/f'{i:06d}.png'),np.full((64,64),600,np.uint16))
    T=np.eye(4);T[:3,3]=[.1,-.2,.3]
    save_extrinsics(session/'extrinsics.json',{'cam2':CameraExtrinsics('cam2',T)})
    config=tmp_path/'config.json';config.write_text(json.dumps({'extrinsics_path':str(tmp_path/'not-used.json')}))
    mask=tmp_path/'mask.png';cv2.imwrite(str(mask),np.ones((64,64),np.uint8)*255)
    output=tmp_path/'out';output.mkdir()
    path=capture_recorded(session,output,'cam2',config,dict(frames=300,stride=3,max_side=128,max_points=16),
                          mask_path=mask,full_recording=True,sample_fps=5)
    with np.load(path) as data:
        assert 12 <= len(data['video']) <= 14
        assert data['timestamps_ns'][0] == timestamps[0]
        assert data['timestamps_ns'][-1]-timestamps[0] > 2_300_000_000
        assert np.min(np.diff(data['timestamps_ns'])) >= 166_000_000
        np.testing.assert_allclose(data['extrinsics'][0],np.linalg.inv(T),atol=1e-6)
        validate_sequence(data)


def test_world_to_image_projection_uses_camera_pose_and_handles_hidden_geometry():
    K=np.array([[100.,0,31.],[0,110.,23.],[0,0,1.]])
    E=np.eye(4);E[:3,3]=[-.1,.2,-.3]
    coords=np.array([[[.12,-.19,.9],[.1,-.2,.1]]])
    pixels=project_world_tracks(coords,K[None],E[None])
    np.testing.assert_allclose(pixels[0][0],[34.333333333,24.833333333])
    assert pixels[0][1] == [None,None]
    json.dumps(pixels,allow_nan=False)
