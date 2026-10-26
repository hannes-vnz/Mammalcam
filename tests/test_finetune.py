import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mammalcam.models.finetune import Crops, class_weights, train_fold, transforms, warmup_cosine


def test_class_weights_are_balanced_and_zero_for_missing():
    w = class_weights(np.array([0, 0, 0, 1]), 3)
    assert w[0] == pytest.approx(4 / (3 * 3)) and w[1] == pytest.approx(4 / 3) and w[2] == 0


def test_warmup_cosine_rises_then_falls_to_zero():
    lr = [warmup_cosine(s, total=100, warmup=10) for s in range(100)]
    assert lr[0] < lr[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lr[10:], lr[11:]))
    assert lr[-1] < 0.01


@pytest.mark.parametrize("train", [True, False])
def test_transforms_give_square_normalised_tensor(train):
    x = transforms(train, size=64)(Image.new("RGB", (100, 60), "white"))
    assert x.shape == torch.Size([3, 64, 64])


def _write_crops(root, n):
    """Red and blue squares: an easy two-class problem."""
    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        label = ["bobcat", "coyote"][i % 2]
        Image.new("RGB", (40, 40), (220, 20, 20) if label == "bobcat" else (20, 20, 220)).save(root / "images" / f"{i}.jpg")
        rows.append({"crop_file": f"{i}.jpg", "label": label})
    return pd.DataFrame(rows)


def test_crops_dataset_returns_tensor_and_label(tmp_path):
    table = _write_crops(tmp_path, 2)
    x, y = Crops(table.crop_file, [0, 1], tmp_path / "images", transforms(False, 32))[1]
    assert x.shape == torch.Size([3, 32, 32]) and y == 1


def test_train_fold_learns_an_easy_problem(tmp_path):
    torch.manual_seed(0)
    table = _write_crops(tmp_path, 32)
    cfg = {"backbone": "resnet18", "pretrained": False, "epochs": 3, "batch_size": 8, "lr": 1e-3, "size": 32,
           "workers": 0, "max_steps": 0, "crops_dir": tmp_path}
    model, history = train_fold(table, table, ["bobcat", "coyote"], cfg, torch.device("cpu"))
    assert len(history) == 3
    assert max(h["val_macro_f1"] for h in history) == pytest.approx(1.0)
