import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mammalcam.models.classify import NO_DETECTION, aligned_proba, image_proba_from_crops, sequence_mean, to_labels

CLASSES = ["bobcat", "coyote", "rabbit"]


def test_aligned_proba_puts_unseen_classes_at_zero():
    x = np.array([[0.0], [1.0], [0.1], [0.9]])
    model = LogisticRegression().fit(x, ["rabbit", "bobcat", "rabbit", "bobcat"])
    proba = aligned_proba(model, x, CLASSES)
    assert proba.shape == (4, 3)
    assert np.all(proba[:, 1] == 0)                     # coyote never seen
    assert np.allclose(proba.sum(axis=1), 1)


def test_image_proba_is_confidence_weighted_and_zero_without_crops():
    crop_proba = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    proba = image_proba_from_crops(crop_proba, ["a", "a", "b"], [0.9, 0.3, 0.5], ["a", "b", "c"])
    assert proba[0] == pytest.approx([0.75, 0.25, 0])
    assert proba[1] == pytest.approx([0, 0, 1])
    assert proba[2] == pytest.approx([0, 0, 0])         # image c had no detection


def test_sequence_mean_fills_frames_without_detection():
    proba = np.array([[1.0, 0, 0], [0, 0, 0], [0.5, 0.5, 0], [0, 0, 1.0]])
    out = sequence_mean(proba, ["s1", "s1", "s1", "s2"])
    assert out[1] == pytest.approx([0.75, 0.25, 0])     # empty frame gets the burst's average
    assert out[0] == pytest.approx(out[2])
    assert out[3] == pytest.approx([0, 0, 1])


def test_to_labels_marks_missing_detection():
    labels = to_labels(np.array([[0.2, 0.7, 0.1], [0, 0, 0]]), CLASSES)
    assert labels.tolist() == ["coyote", NO_DETECTION]
