import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mammalcam.data.cct import DEFAULT_ANN_DIR, EMPTY, MULTI, image_label, load_cct
from mammalcam.data.split import SPLITS, build_all, location_split, overlap, random_split


# --- label mapping ---------------------------------------------------------

@pytest.mark.parametrize("cats, expected", [
    (["coyote"], "coyote"),
    (["coyote", "coyote"], "coyote"),          # two coyotes are still one class
    (["bobcat", "rabbit"], MULTI),
    (["empty"], EMPTY),
    (["empty", "deer"], "deer"),               # empty never beats an animal
    ([], EMPTY),
])
def test_image_label(cats, expected):
    assert image_label(cats) == expected


def _write_coco(path: Path, images, annotations, categories):
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))


def _img(i, loc, seq):
    return {"id": f"im{i}", "file_name": f"im{i}.jpg", "location": loc, "seq_id": seq, "frame_num": 0,
            "seq_num_frames": 1, "date_captured": "2012-01-01 00:00:00", "width": 100, "height": 80}


def test_load_cct_merges_files_and_maps_labels(tmp_path):
    cats = [{"id": 1, "name": "coyote"}, {"id": 2, "name": "rabbit"}, {"id": 30, "name": "empty"}]
    _write_coco(tmp_path / "train_annotations.json",
                [_img(1, 10, "s1"), _img(2, 10, "s1")],
                [{"id": "a1", "image_id": "im1", "category_id": 1, "bbox": [1, 2, 3, 4]},
                 {"id": "a2", "image_id": "im1", "category_id": 2, "bbox": [5, 6, 7, 8]},
                 {"id": "a3", "image_id": "im2", "category_id": 30}], cats)
    _write_coco(tmp_path / "trans_test_annotations.json",
                [_img(3, 20, "s2")],
                [{"id": "a4", "image_id": "im3", "category_id": 1}], cats)

    img, ann = load_cct(tmp_path)
    img = img.set_index("image_id")
    assert img.loc["im1", "label"] == MULTI and img.loc["im1", "n_boxes"] == 2
    assert img.loc["im2", "label"] == EMPTY and img.loc["im2", "n_boxes"] == 0
    assert img.loc["im3", "label"] == "coyote" and img.loc["im3", "benchmark_split"] == "trans_test"
    assert ann.set_index("annotation_id").loc["a1", "bbox"] == (1, 2, 3, 4)


def test_load_cct_rejects_annotations_for_unknown_images(tmp_path):
    _write_coco(tmp_path / "train_annotations.json", [_img(1, 10, "s1")],
                [{"id": "a1", "image_id": "nope", "category_id": 1}], [{"id": 1, "name": "coyote"}])
    with pytest.raises(ValueError, match="unknown images"):
        load_cct(tmp_path)


# --- splits ----------------------------------------------------------------

@pytest.fixture
def synthetic():
    """12 locations, 3-frame sequences, classes spread over several locations."""
    rng = np.random.default_rng(1)
    rows = []
    classes = ["coyote", "rabbit", "bobcat", "deer", EMPTY]
    for loc in range(12):
        local = [classes[(loc + k) % len(classes)] for k in range(3)]
        for s in range(rng.integers(5, 30)):
            label = local[s % 3]
            for f in range(3):
                rows.append({"image_id": f"{loc}-{s}-{f}", "location": loc, "seq_id": f"{loc}-{s}", "label": label})
    return pd.DataFrame(rows)


def test_location_split_has_no_shared_location(synthetic):
    assign = location_split(synthetic, n_trials=200)
    split = synthetic.location.map(assign)
    assert set(split) == set(SPLITS)
    assert overlap(synthetic, split, "location")["n_shared"] == 0
    assert overlap(synthetic, split, "seq_id")["n_shared"] == 0


def test_location_split_covers_classes_in_train_and_test(synthetic):
    split = synthetic.location.map(location_split(synthetic, n_trials=200))
    for part in ("train", "test"):
        assert set(synthetic.label[split == part]) == set(synthetic.label)


def test_location_split_is_deterministic(synthetic):
    assert location_split(synthetic, seed=3, n_trials=100) == location_split(synthetic, seed=3, n_trials=100)


def test_random_image_split_matches_counts_and_leaks(synthetic):
    counts = {"train": len(synthetic) - 40, "val": 15, "test": 25}
    split = random_split(synthetic, counts, by="image", seed=0)
    assert split.value_counts().to_dict() == counts
    assert overlap(synthetic, split, "seq_id")["n_shared"] > 0  # the point of this baseline


def test_random_sequence_split_keeps_sequences_together(synthetic):
    counts = {"train": len(synthetic) - 60, "val": 24, "test": 36}
    split = random_split(synthetic, counts, by="sequence", seed=0)
    assert overlap(synthetic, split, "seq_id")["n_shared"] == 0
    assert set(split) == set(SPLITS)


def test_build_all_uses_equal_sizes(synthetic):
    _, splits = build_all(synthetic, seed=0)
    loc = splits["location"].value_counts()
    assert splits["random_image"].value_counts().to_dict() == loc.to_dict()
    # sequence split can only be off by less than one sequence per boundary
    assert (splits["random_sequence"].value_counts() - loc).abs().max() < 3 * 2


# --- real data (skipped when CCT20 is not downloaded) ----------------------

@pytest.mark.skipif(not DEFAULT_ANN_DIR.exists(), reason="CCT20 not downloaded")
def test_cct20_location_split_from_config():
    img, _ = load_cct()
    cfg = json.loads(Path("configs/splits_cct20.json").read_text())
    locs = [loc for part in SPLITS for loc in cfg["locations"][part]]
    assert len(locs) == len(set(locs)) == img.location.nunique()
    assign = {loc: part for part in SPLITS for loc in cfg["locations"][part]}
    assert location_split(img, seed=cfg["seed"], val_frac=cfg["val_frac"], test_frac=cfg["test_frac"]) == assign


# --- camera cross-validation ------------------------------------------------

def test_camera_folds_test_every_camera_once_and_keep_classes_in_train(synthetic):
    from mammalcam.data.split import camera_folds, fold_parts

    k = 4
    folds = camera_folds(synthetic, k=k, n_trials=300)
    assert set(folds) == set(synthetic.location) and set(folds.values()) == set(range(k))
    tested = []
    for f in range(k):
        parts = fold_parts(folds, f, k)
        assert set(parts.values()) == set(SPLITS)
        tested += [loc for loc, p in parts.items() if p == "test"]
        part = synthetic.location.map(parts)
        assert set(synthetic.label[part == "train"]) == set(synthetic.label)
    assert sorted(tested) == sorted(folds)  # each camera tested exactly once


def test_camera_folds_are_deterministic(synthetic):
    from mammalcam.data.split import camera_folds

    assert camera_folds(synthetic, k=3, seed=1, n_trials=100) == camera_folds(synthetic, k=3, seed=1, n_trials=100)
