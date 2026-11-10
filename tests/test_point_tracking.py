from __future__ import annotations

import json

import numpy as np
import pytest

from anatomy_guided_recon3d.config import CameraIntrinsics, DepthConfig, RGBDFrame
from anatomy_guided_recon3d.tracking.sequence import prepare_frame, save_sequence, seed_queries, validate_sequence
from anatomy_guided_recon3d.tracking.torch_knn import knnquery
from anatomy_guided_recon3d.tracking.worker import make_preview


def frame(timestamp=1_000_000_000, width=256, height=192):
    return RGBDFrame("cam2", np.full((height, width, 3), [170, 110, 95], np.uint8),
                     np.full((height, width), 300, np.uint16), timestamp,
                     CameraIntrinsics(width, height, 200., 205., width/2-.5, height/2-.5),
                     DepthConfig(.002, .25, 1.2))


def test_metric_depth_and_inverse_calibration_round_trip(tmp_path):
    T = np.array([[0., -1., 0., .2], [1., 0., 0., -.1], [0., 0., 1., .3], [0., 0., 0., 1.]])
    f = frame()
    mask = np.zeros(f.depth_mm.shape, bool)
    mask[40:140, 60:190] = True
    prepared = prepare_frame(f, mask, T)
    np.testing.assert_allclose(prepared["depths"], .6)
    np.testing.assert_allclose(prepared["extrinsics"], np.linalg.inv(T), atol=1e-6)
    queries = seed_queries(prepared)
    assert 100 <= len(queries) <= 128
    camera = queries[:, 1:] @ prepared["extrinsics"][:3, :3].T + prepared["extrinsics"][:3, 3]
    uv = camera @ prepared["intrinsics"].T
    uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
    assert prepared["mask"][uv[:, 1], uv[:, 0]].all()
    np.testing.assert_allclose(camera[:, 2], .6, atol=1e-6)
    second = dict(prepared, timestamp_ns=f.timestamp_ns+100_000_000)
    path = tmp_path / "sequence.npz"
    save_sequence(path, [prepared, second], "cam2")
    with np.load(path, allow_pickle=False) as data:
        assert validate_sequence(data) == (2, len(queries))
        assert str(data["length_unit"]) == "metre"
        assert str(data["camera_id"]) == "cam2"


def test_resized_intrinsics_keep_pixel_centres_and_depth_units():
    f = frame(width=1280, height=720)
    prepared = prepare_frame(f, np.ones(f.depth_mm.shape, bool), np.eye(4), 512)
    assert prepared["video"].shape == (288, 512, 3)
    np.testing.assert_allclose(prepared["intrinsics"][:2, 2], [255.5, 143.5])
    np.testing.assert_allclose(prepared["intrinsics"][[0, 1], [0, 1]], [80, 82])
    np.testing.assert_allclose(prepared["depths"], .6)


def test_invalid_depth_and_non_temporal_inputs_are_rejected(tmp_path):
    f = frame()
    prepared = prepare_frame(f, np.zeros(f.depth_mm.shape, bool), np.eye(4))
    with pytest.raises(ValueError, match="No valid interior"):
        seed_queries(prepared)
    with pytest.raises(ValueError, match="strictly increasing"):
        save_sequence(tmp_path / "bad.npz", [prepared, prepared], "cam2")


def test_knn_never_crosses_batch_boundaries():
    import torch
    context = torch.tensor([[0., 0, 0], [2., 0, 0], [100., 0, 0], [102., 0, 0]])
    query = torch.tensor([[1.9, 0, 0], [100.1, 0, 0]])
    idx, distance = knnquery(2, context, query, torch.tensor([2, 4]), torch.tensor([1, 2]))
    assert idx.dtype == torch.int32
    assert idx.tolist() == [[1, 0], [2, 3]]
    np.testing.assert_allclose(distance.numpy(), [[.1, 1.9], [.1, 1.9]], atol=1e-5)


def test_preview_marks_nonfinite_predictions_hidden_and_keeps_point_ids():
    data = {"depths":np.ones((2, 2, 2), np.float32), "video":np.zeros((2, 2, 2, 3), np.uint8),
            "intrinsics":np.tile(np.eye(3), (2, 1, 1)), "extrinsics":np.tile(np.eye(4), (2, 1, 1)),
            "query_point":np.array([[0, 0, 0, 1], [0, 1, 0, 1]]), "timestamps_ns":np.array([10, 20])}
    coords = np.array([[[0, 0, 1], [1, 0, 1]], [[.001, 0, 1], [np.nan, 0, 1]]])
    result = make_preview(data, coords, np.ones((2, 2), bool))
    assert result["point_count"] == 2
    assert result["coords"][1][1] == [None, None, None]
    assert result["visible"] == [[True, True], [True, False]]
    assert result["displacement_mm"][1][0] == 1
    json.dumps(result, allow_nan=False)
