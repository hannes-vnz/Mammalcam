"""Crop classifier vs. full-image baseline, evaluated end to end per image.

Usage:
    python -m mammalcam.models.classify [--folds 5]

Both models are linear probes on the same frozen backbone. They differ only
in their input:

- full: the whole image (as in mammalcam.models.baseline)
- crops: MegaDetector crops. Trained on the labelled crops (crops.csv); at
  test time every detection of an image is classified (detections.csv, no
  ground-truth boxes involved) and the crop probabilities are averaged,
  weighted by detection confidence. An image without a detection gets the
  prediction "no detection" and counts as wrong.

Test images are the same for both: images with exactly one of the classifier
classes. Predictions are also averaged over each burst (sequence), which
fills in frames where the animal has already left.

Evaluations:
- 5-fold cross-validation over camera groups (main result; every camera is
  tested exactly once)
- the single location split and the two random splits (overestimation)

Writes reports/05_crop_classifier.md, reports/results/classify.json and
data/predictions/classify_test.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mammalcam.data.cct import DEFAULT_ANN_DIR, DEFAULT_IMG_DIR, load_cct
from mammalcam.data.split import camera_folds, fold_parts
from mammalcam.eval import metrics
from mammalcam.models import features, linear_probe
from mammalcam.models.baseline import STRATEGIES, load_classes, select_images

NO_DETECTION = "no detection"
CROPS_DIR = Path("data/crops")


def aligned_proba(model, x, classes: list[str]) -> np.ndarray:
    """predict_proba with columns in ``classes`` order (zeros for classes the model never saw)."""
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = proba[:, j]
    return out


def image_proba_from_crops(crop_proba: np.ndarray, crop_image_ids, weights, image_ids) -> np.ndarray:
    """Confidence-weighted mean of crop probabilities per image; zero rows for images without crops."""
    df = pd.DataFrame(crop_proba * np.asarray(weights)[:, None])
    df["image_id"] = np.asarray(crop_image_ids)
    df["w"] = np.asarray(weights)
    sums = df.groupby("image_id").sum()
    probs = sums.drop(columns="w").div(sums.w, axis=0)
    return probs.reindex(image_ids).fillna(0.0).to_numpy()


def sequence_mean(proba: np.ndarray, seq_ids) -> np.ndarray:
    """Average over the frames of each sequence that have a prediction; every frame gets the result."""
    df = pd.DataFrame(proba)
    has = proba.sum(axis=1) > 0
    df["seq"] = np.asarray(seq_ids)
    means = df[has].groupby("seq").mean()
    return means.reindex(df.seq).fillna(0.0).to_numpy()


def to_labels(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    labels = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
    labels[proba.sum(axis=1) == 0] = NO_DETECTION
    return labels


class Data:
    """Everything both models need, loaded once."""

    def __init__(self, ann_dir: Path, img_dir: Path, workers: int):
        self.classes = load_classes()
        images, _ = load_cct(ann_dir)
        self.images = images
        self.table = select_images(images, self.classes)  # test images for both models
        self.x_full = features.cached("full_images", self.table, "file_name", img_dir, workers=workers)
        self.dets = pd.read_csv(CROPS_DIR / "detections.csv")
        self.x_dets = features.cached("crops", self.dets, "crop_file", CROPS_DIR / "images", workers=workers)
        crops = pd.read_csv(CROPS_DIR / "crops.csv")
        crops = crops[crops.label.isin(self.classes)]
        row_of = pd.Series(np.arange(len(self.dets)), index=self.dets.crop_file)
        self.train_crops = crops.assign(row=crops.crop_file.map(row_of).values)


def evaluate(data: Data, part_of_location_or_image: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Train both models on one train/val/test assignment and predict the test images.

    part_of_location_or_image: split name per image_id.
    Returns (predictions per test image, chosen C per model).
    """
    classes, t = data.classes, data.table
    part = t.image_id.map(part_of_location_or_image).values
    test = t[part == "test"]
    out = test[["image_id", "seq_id", "location", "label"]].copy()
    chosen = {}

    # full images
    y = t.label.values
    m, _ = linear_probe.fit(data.x_full[part == "train"], y[part == "train"],
                            data.x_full[part == "val"], y[part == "val"])
    proba = aligned_proba(m, data.x_full[part == "test"], classes)
    out["full_image"] = to_labels(proba, classes)
    out["full_sequence"] = to_labels(sequence_mean(proba, test.seq_id), classes)
    chosen["full"] = float(m[-1].C)

    # crops: train and validate on labelled crops, test on all detections of the test images
    tc = data.train_crops
    cpart = tc.image_id.map(part_of_location_or_image).values
    xc, yc = data.x_dets[tc.row.values], tc.label.values
    m, _ = linear_probe.fit(xc[cpart == "train"], yc[cpart == "train"], xc[cpart == "val"], yc[cpart == "val"])
    d = data.dets[data.dets.image_id.isin(test.image_id)]
    crop_proba = aligned_proba(m, data.x_dets[d.index.values], classes)
    proba = image_proba_from_crops(crop_proba, d.image_id, d.conf, test.image_id)
    out["crops_image"] = to_labels(proba, classes)
    out["crops_sequence"] = to_labels(sequence_mean(proba, test.seq_id), classes)
    chosen["crops"] = float(m[-1].C)
    return out, chosen


PRED_COLUMNS = ("full_image", "full_sequence", "crops_image", "crops_sequence")


def scores(preds: pd.DataFrame, classes: list[str]) -> dict:
    return {col: metrics.summary(preds.label, preds[col], classes) for col in PRED_COLUMNS}


def build_report(res: dict, cv_preds: pd.DataFrame, classes: list[str], backbone: str, folds: dict) -> str:
    names = {"full_image": "full image", "full_sequence": "full image + burst average",
             "crops_image": "MegaDetector crops", "crops_sequence": "MegaDetector crops + burst average"}

    cv = res["cv"]
    rows = {}
    for col in PRED_COLUMNS:
        acc = [f[col]["accuracy"] for f in cv["folds"]]
        f1 = [f[col]["macro_f1"] for f in cv["folds"]]
        rows[names[col]] = {
            "accuracy (pooled)": cv["pooled"][col]["accuracy"],
            "macro-F1 (pooled)": cv["pooled"][col]["macro_f1"],
            "accuracy per fold": f"{np.mean(acc):.3f} ± {np.std(acc):.3f}",
            "macro-F1 per fold": f"{np.mean(f1):.3f} ± {np.std(f1):.3f}",
        }
    single = pd.DataFrame({s: {names[c]: res["single"][s]["scores"][c]["accuracy"] for c in PRED_COLUMNS}
                           for s in STRATEGIES})
    single_f1 = pd.DataFrame({s: {names[c]: res["single"][s]["scores"][c]["macro_f1"] for c in PRED_COLUMNS}
                              for s in STRATEGIES})

    no_det = (cv_preds.crops_image == NO_DETECTION).mean()
    per_class = pd.concat({
        "full image F1": metrics.per_class(cv_preds.label, cv_preds.full_image, classes).f1,
        "crops F1": metrics.per_class(cv_preds.label, cv_preds.crops_image, classes).f1,
        "crops + burst F1": metrics.per_class(cv_preds.label, cv_preds.crops_sequence, classes).f1,
        "support": metrics.per_class(cv_preds.label, cv_preds.crops_image, classes).support,
    }, axis=1)
    per_cam = cv_preds.assign(
        full=cv_preds.label == cv_preds.full_image,
        crops=cv_preds.label == cv_preds.crops_image,
        crops_burst=cv_preds.label == cv_preds.crops_sequence,
    ).groupby("location")[["full", "crops", "crops_burst"]].mean()
    per_cam.insert(0, "images", cv_preds.groupby("location").size())
    per_cam.insert(0, "fold", per_cam.index.map(folds))

    lines = [
        "# Crop classifier vs. full-image baseline", "",
        "Generated by `python -m mammalcam.models.classify`. Do not edit by hand.", "",
        f"- Both models: frozen `{backbone}` embeddings + logistic regression (balanced class weights, "
        "C chosen on the validation cameras)",
        "- Full image: whole image resized to 224×224",
        "- Crops: trained on labelled MegaDetector crops; tested on **every** MegaDetector animal detection "
        "(conf ≥ 0.2) of the test image, no ground-truth boxes involved. Crop probabilities are averaged per image, "
        "weighted by detection confidence. No detection → counted as wrong.",
        "- Burst average: probabilities averaged over the frames of a burst (sequence).",
        f"- Test images: {len(cv_preds):,} images with exactly one of the {len(classes)} classes; "
        f"{no_det:.1%} of them have no MegaDetector detection.", "",
        f"## Main result: {len(cv['folds'])}-fold cross-validation over camera groups", "",
        "Every camera is in the test group exactly once. Pooled = all test predictions together; "
        "per fold = mean ± standard deviation over the folds.", "",
        pd.DataFrame(rows).T.to_markdown(floatfmt=".3f"), "",
        "## Single splits: how much a random split overestimates (accuracy)", "",
        single.to_markdown(floatfmt=".3f"), "",
        "Macro-F1:", "",
        single_f1.to_markdown(floatfmt=".3f"), "",
        "## Per class (cross-validation, pooled)", "",
        per_class.sort_values("support", ascending=False).to_markdown(floatfmt=".3f"), "",
        "## Accuracy per camera (cross-validation)", "",
        per_cam.to_markdown(floatfmt=".3f"), "",
        "## Confusion matrix: crops + burst average (cross-validation, rows true)", "",
        metrics.confusion(cv_preds.label, cv_preds.crops_sequence, classes).to_markdown(), "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    data = Data(args.ann_dir, args.img_dir, args.workers)
    classes = data.classes
    res = {"single": {}, "cv": {"folds": []}}
    all_preds = []

    for name in STRATEGIES:
        split = pd.read_csv(args.splits / f"{name}.csv").set_index("image_id").split
        preds, chosen = evaluate(data, split)
        res["single"][name] = {"scores": scores(preds, classes), "C": chosen}
        all_preds.append(preds.assign(evaluation=name))
        print(f"{name:16s}", {k: round(v["accuracy"], 3) for k, v in res["single"][name]["scores"].items()})

    folds = camera_folds(data.images, args.folds, args.seed)
    res["cv"]["camera_groups"] = {str(f): [loc for loc, g in folds.items() if g == f] for f in range(args.folds)}
    cv_preds = []
    for f in range(args.folds):
        loc_part = fold_parts(folds, f, args.folds)
        split = data.images.set_index("image_id").location.map(loc_part)
        preds, chosen = evaluate(data, split)
        res["cv"]["folds"].append({**scores(preds, classes), "C": chosen})
        cv_preds.append(preds.assign(evaluation=f"cv_fold_{f}"))
        print(f"fold {f}", {k: round(v["accuracy"], 3) for k, v in scores(preds, classes).items()})
    cv_all = pd.concat(cv_preds)
    res["cv"]["pooled"] = scores(cv_all, classes)

    Path("reports/results").mkdir(parents=True, exist_ok=True)
    Path("reports/results/classify.json").write_text(json.dumps(
        {"backbone": features.DEFAULT_BACKBONE, "classes": classes, **res}, indent=2))
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    pd.concat(all_preds + cv_preds).to_csv("data/predictions/classify_test.csv", index=False)
    Path("reports/05_crop_classifier.md").write_text(
        build_report(res, cv_all, classes, features.DEFAULT_BACKBONE, folds))
    print("wrote reports/05_crop_classifier.md")


if __name__ == "__main__":
    main()
