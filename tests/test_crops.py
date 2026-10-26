from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from mammalcam.crops.make_crops import _crop_one, build_crop_table, label_detections, square_box


# --- crop geometry -----------------------------------------------------------

def test_square_box_pads_and_centres():
    # 100x50 px box in the middle of a 1000x1000 image, 10% padding per side
    left, top, right, bottom = square_box((0.45, 0.475, 0.1, 0.05), 1000, 1000, pad=0.1)
    assert right - left == bottom - top == 120
    assert (left + right) / 2 == pytest.approx(500, abs=1)
    assert (top + bottom) / 2 == pytest.approx(500, abs=1)


def test_square_box_is_shifted_inside_at_the_edge():
    left, top, right, bottom = square_box((0.0, 0.9, 0.1, 0.1), 1000, 1000, pad=0.1)
    assert left == 0 and bottom == 1000
    assert right - left == bottom - top == 120


def test_square_box_is_clipped_when_larger_than_image():
    left, top, right, bottom = square_box((0.0, 0.0, 1.0, 0.5), 400, 200, pad=0.1)
    assert (left, top, right, bottom) == (0, 0, 400, 200)


# --- label assignment --------------------------------------------------------

def _row(label):
    return SimpleNamespace(label=label, width=1000, height=1000)


DET = ("animal", 0.9, (0.1, 0.1, 0.2, 0.2))
ROCK = ("animal", 0.5, (0.7, 0.7, 0.1, 0.1))


def test_single_species_with_boxes_keeps_only_matching_detections():
    crops = label_detections(_row("coyote"), [("coyote", (0.1, 0.1, 0.2, 0.2))], True, [DET, ROCK])
    assert [(c["label"], c["source"]) for c in crops] == [("coyote", "box_match")]


def test_single_species_without_boxes_uses_image_label():
    crops = label_detections(_row("coyote"), [], False, [DET, ROCK])
    assert [(c["label"], c["source"]) for c in crops] == [("coyote", "image_label")] * 2


def test_multi_species_takes_label_from_matching_box():
    boxes = [("bobcat", (0.1, 0.1, 0.2, 0.2)), ("rabbit", (0.7, 0.7, 0.1, 0.1))]
    crops = label_detections(_row("multi"), boxes, True, [DET, ROCK, ("animal", 0.4, (0.4, 0.4, 0.1, 0.1))])
    assert sorted(c["label"] for c in crops) == ["bobcat", "rabbit"]


def test_each_box_is_used_once():
    duplicate = ("animal", 0.8, (0.11, 0.11, 0.2, 0.2))
    crops = label_detections(_row("multi"), [("bobcat", (0.1, 0.1, 0.2, 0.2))], True, [DET, duplicate])
    assert len(crops) == 1 and crops[0]["conf"] == 0.9


@pytest.mark.parametrize("label", ["empty", "car"])
def test_no_crops_for_empty_and_car(label):
    assert label_detections(_row(label), [], False, [DET]) == []


def test_threshold_and_category_filter():
    dets = [("animal", 0.1, (0.1, 0.1, 0.2, 0.2)), ("person", 0.9, (0.1, 0.1, 0.2, 0.2))]
    assert label_detections(_row("coyote"), [], False, dets, threshold=0.2) == []


# --- table and files ---------------------------------------------------------

def test_build_crop_table_drops_tiny_detections():
    images = pd.DataFrame([{"image_id": "a", "file_name": "a.jpg", "location": 1, "seq_id": "s",
                            "label": "rodent", "width": 2048, "height": 1536}])
    ann = pd.DataFrame([{"image_id": "a", "category": "rodent", "bbox": None}])
    dets = {"a.jpg": [("animal", 0.9, (0.1, 0.1, 0.2, 0.2)),        # ~205 px in the 1024 px image
                      ("animal", 0.9, (0.5, 0.5, 0.005, 0.005))]}   # ~5 px
    crops = build_crop_table(images, ann, dets, 0.2)
    assert len(crops) == 1 and crops.bbox[0] == (0.1, 0.1, 0.2, 0.2)


def test_crop_one_writes_resized_square(tmp_path):
    src = tmp_path / "img.jpg"
    Image.new("RGB", (1024, 768), "white").save(src)
    out = tmp_path / "crops"
    out.mkdir()
    written = _crop_one((src, [("c0.jpg", (0.25, 0.25, 0.5, 0.5))], out, 128))
    assert written == ["c0.jpg"]
    with Image.open(out / "c0.jpg") as im:
        assert max(im.size) == 128
