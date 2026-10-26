import json

import pandas as pd
import pytest

from mammalcam.data.sample import stratified_sample
from mammalcam.detect.evaluate import ground_truth, iou, load_detections, match, to_relative


# --- box geometry ----------------------------------------------------------

def test_to_relative_uses_original_size():
    # CCT20 boxes are in pixels of the original 2048x1494 image, not the 1024 px download
    assert to_relative((1024, 747, 512, 373.5), 2048, 1494) == pytest.approx((0.5, 0.5, 0.25, 0.25))


@pytest.mark.parametrize("a, b, expected", [
    ((0, 0, 1, 1), (0, 0, 1, 1), 1.0),
    ((0, 0, 1, 1), (2, 2, 1, 1), 0.0),
    ((0, 0, 2, 1), (1, 0, 2, 1), 1 / 3),
    ((0, 0, 1, 1), (0.25, 0.25, 0.5, 0.5), 0.25),
])
def test_iou(a, b, expected):
    assert iou(a, b) == pytest.approx(expected)
    assert iou(b, a) == pytest.approx(expected)


def test_match_is_one_to_one_and_prefers_confident_detections():
    gt = [(0, 0, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)]
    det = [
        (0.3, (0.01, 0.01, 0.2, 0.2)),   # duplicate of the first box, lower confidence
        (0.9, (0, 0, 0.2, 0.2)),
        (0.8, (0.8, 0.0, 0.1, 0.1)),     # nothing there
    ]
    ious, fp, fn = match(gt, det)
    assert ious == [pytest.approx(1.0)]
    assert fp == 2 and fn == 1


def test_match_respects_iou_threshold():
    ious, fp, fn = match([(0, 0, 1, 1)], [(0.9, (0, 0, 0.4, 1))], iou_threshold=0.5)
    assert ious == [] and fp == 1 and fn == 1


# --- ground truth ----------------------------------------------------------

def test_ground_truth_flags_missing_and_full_frame_boxes():
    images = pd.DataFrame({"image_id": ["a", "b", "c"], "width": [200, 200, 200], "height": [100, 100, 100]})
    ann = pd.DataFrame({
        "image_id": ["a", "b", "b", "c", "c"],
        "category": ["coyote", "car", "empty", "bobcat", "rabbit"],
        "bbox": [(20, 10, 40, 20), (0, 0, 200, 100), None, (0, 0, 20, 10), None],
    })
    gt = ground_truth(images, ann)
    assert gt["a"] == {"boxes": [pytest.approx((0.1, 0.1, 0.2, 0.2))], "complete": True}
    assert gt["b"]["complete"] is False and gt["b"]["boxes"] == []   # full-frame car box
    assert gt["c"]["complete"] is False and len(gt["c"]["boxes"]) == 1  # rabbit has no box


def test_load_detections_maps_categories(tmp_path):
    path = tmp_path / "md.json"
    path.write_text(json.dumps({
        "detection_categories": {"1": "animal", "3": "vehicle"},
        "images": [{"file": "/some/dir/x.jpg", "detections": [{"category": "1", "conf": 0.7, "bbox": [0, 0, 1, 1]}]}],
    }))
    assert load_detections(path) == {"x.jpg": [("animal", 0.7, (0, 0, 1, 1))]}


# --- sample ----------------------------------------------------------------

def test_stratified_sample_is_balanced_and_one_per_sequence():
    rows = [{"image_id": f"{lab}{s}{f}", "seq_id": f"{lab}{s}", "label": lab}
            for lab, n_seq in [("common", 50), ("mid", 20), ("rare", 3)]
            for s in range(n_seq) for f in range(3)]
    images = pd.DataFrame(rows)
    sample = stratified_sample(images, n=30, seed=0)
    assert len(sample) == 30
    assert sample.seq_id.is_unique
    counts = sample.label.value_counts()
    assert counts["rare"] == 3                          # everything the rare label has
    assert sorted([counts["common"], counts["mid"]]) == [13, 14]  # the other 27 split evenly
