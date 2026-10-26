import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mammalcam.eval import metrics
from mammalcam.models import linear_probe
from mammalcam.models.baseline import load_classes, run_strategy, select_images
from mammalcam.models.features import ImageFiles, cached

CLASSES = ["bobcat", "coyote", "rabbit"]


# --- class selection ---------------------------------------------------------

def test_class_config_excludes_rare_and_stage1_labels():
    classes = load_classes()
    assert len(classes) == len(set(classes)) == 12
    assert not {"fox", "badger", "empty", "car", "multi"} & set(classes)


def test_select_images_keeps_only_classifier_classes():
    images = pd.DataFrame({"image_id": list("abcdef"),
                           "label": ["coyote", "empty", "multi", "fox", "rabbit", "car"]})
    assert select_images(images, CLASSES).label.tolist() == ["coyote", "rabbit"]


# --- metrics -----------------------------------------------------------------

def test_summary_and_per_class():
    y_true = ["bobcat", "bobcat", "coyote", "rabbit"]
    y_pred = ["bobcat", "coyote", "coyote", "rabbit"]
    s = metrics.summary(y_true, y_pred, CLASSES)
    assert s["n"] == 4 and s["accuracy"] == 0.75
    pc = metrics.per_class(y_true, y_pred, CLASSES)
    assert pc.loc["bobcat", "recall"] == 0.5 and pc.loc["coyote", "precision"] == 0.5
    assert s["macro_f1"] == pytest.approx(pc.f1.mean())


def test_prediction_outside_classes_counts_as_wrong_and_shows_in_confusion():
    y_true, y_pred = ["bobcat", "coyote"], ["no detection", "coyote"]
    assert metrics.summary(y_true, y_pred, CLASSES)["accuracy"] == 0.5
    cm = metrics.confusion(y_true, y_pred, CLASSES)
    assert list(cm.columns) == CLASSES + ["no detection"]
    assert cm.loc["bobcat", "no detection"] == 1


# --- linear probe ------------------------------------------------------------

@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, (3, 8))
    y = np.repeat(CLASSES, 40)
    x = centers[np.repeat(range(3), 40)] + rng.normal(0, 1, (120, 8))
    return x, y


def test_linear_probe_learns_separable_classes(blobs):
    x, y = blobs
    model, scores = linear_probe.fit(x[::2], y[::2], x[1::2], y[1::2])
    assert set(scores) == set(linear_probe.C_GRID)
    assert (model.predict(x[1::2]) == y[1::2]).mean() > 0.95


def test_run_strategy_only_tests_on_test_part(blobs):
    x, y = blobs
    table = pd.DataFrame({"image_id": range(len(y)), "location": 0, "label": y})
    split = pd.Series(np.tile(["train", "train", "val", "test"], len(y) // 4))
    results, preds = run_strategy(x, table, split, CLASSES)
    assert results["test"]["n"] == len(preds) == (split == "test").sum()
    assert set(preds.image_id) == set(table.image_id[(split == "test").values])


# --- features ----------------------------------------------------------------

def test_image_files_resizes_without_crop(tmp_path):
    path = tmp_path / "wide.jpg"
    Image.new("RGB", (400, 100), (255, 0, 0)).save(path)
    x = ImageFiles([path], 32, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))[0]
    assert x.shape == torch.Size([3, 32, 32])
    assert x[0].mean() > 0.9  # whole image kept, still red after normalisation


def test_cached_reuses_embeddings(tmp_path, monkeypatch):
    calls = []

    def fake_extract(paths, backbone, **kwargs):
        calls.append(len(paths))
        return np.ones((len(paths), 4), dtype=np.float32)

    monkeypatch.setattr("mammalcam.models.features.extract", fake_extract)
    table = pd.DataFrame({"file": ["a.jpg", "b.jpg"]})
    first = cached("t", table, "file", tmp_path, cache_dir=tmp_path / "cache")
    second = cached("t", table, "file", tmp_path, cache_dir=tmp_path / "cache")
    assert calls == [2] and np.array_equal(first, second)
    cached("t", table.iloc[:1], "file", tmp_path, cache_dir=tmp_path / "cache")  # different inputs: recompute
    assert calls == [2, 1]
