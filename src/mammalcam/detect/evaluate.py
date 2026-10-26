"""Compare MegaDetector output with the CCT20 ground-truth boxes.

Usage:
    python -m mammalcam.detect.evaluate [--detections data/detections/md_sample.json]

Two questions:
1. Image level: does MegaDetector find an animal on images that have one,
   and nothing on empty images? (This is the empty filter, stage 1.)
2. Box level: do its animal boxes overlap the annotated ones (IoU)?

Ground-truth boxes are stored in pixels of the original image size, while
the downloaded images are downscaled. Both are converted to [x, y, w, h]
relative to the image, the format MegaDetector writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mammalcam.data.cct import DEFAULT_ANN_DIR, EMPTY, load_cct
from mammalcam.detect.megadetector import MD_CATEGORIES

LILA_RESULTS = Path("data/raw/cct20/megadetector/caltech-camera-traps_mdv5a.0.0_results.json")

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
BOX_THRESHOLD = 0.2  # MegaDetector v5's typical threshold
IOU_THRESHOLD = 0.5
VEHICLE_CLASSES = {"car"}
FULL_FRAME = 0.9  # boxes covering more than this share of the image are not real boxes


def to_relative(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel [x, y, w, h] in the original image size -> relative [x, y, w, h]."""
    x, y, w, h = bbox
    return (x / width, y / height, w / width, h / height)


def iou(a, b) -> float:
    """Intersection over union of two [x, y, w, h] boxes."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(gt: list, det: list, iou_threshold: float = IOU_THRESHOLD) -> tuple[list[float], int, int]:
    """Greedy one-to-one matching, highest-confidence detection first.

    gt: list of boxes; det: list of (confidence, box).
    Returns (IoUs of the matched pairs, unmatched detections, unmatched gt).
    """
    free = list(range(len(gt)))
    ious = []
    unmatched_det = 0
    for _, box in sorted(det, key=lambda d: -d[0]):
        scores = [(iou(box, gt[i]), i) for i in free]
        best = max(scores, default=(0.0, None))
        if best[1] is not None and best[0] >= iou_threshold:
            ious.append(best[0])
            free.remove(best[1])
        else:
            unmatched_det += 1
    return ious, unmatched_det, len(free)


def load_detections(path: Path) -> dict[str, list[tuple[str, float, tuple]]]:
    """file name -> [(category name, confidence, relative box)]."""
    data = json.loads(Path(path).read_text())
    cats = data.get("detection_categories", MD_CATEGORIES)
    out = {}
    for im in data["images"]:
        if im.get("failure"):
            raise RuntimeError(f"MegaDetector failed on {im['file']}: {im['failure']}")
        out[Path(im["file"]).name] = [(cats[d["category"]], d["conf"], tuple(d["bbox"])) for d in im["detections"]]
    return out


def ground_truth(images: pd.DataFrame, ann: pd.DataFrame) -> dict[str, dict]:
    """image_id -> {"boxes": relative animal boxes, "complete": every animal has a real box}."""
    meta = images.set_index("image_id")
    out = {}
    for image_id, group in ann[ann.category != EMPTY].groupby("image_id"):
        w, h = meta.loc[image_id, ["width", "height"]]
        boxes, complete = [], True
        for bbox in group.bbox:
            if bbox is None:
                complete = False
                continue
            rel = to_relative(bbox, w, h)
            if rel[2] * rel[3] > FULL_FRAME:
                complete = False
                continue
            boxes.append(rel)
        out[image_id] = {"boxes": boxes, "complete": complete}
    return out


def image_level(sample: pd.DataFrame, dets: dict) -> pd.DataFrame:
    """Per threshold: share of animal images kept, share of empty images dropped."""
    def max_conf(file_name, category):
        return max((c for cat, c, _ in dets[file_name] if cat == category), default=0.0)

    animal = sample[~sample.label.isin({EMPTY} | VEHICLE_CLASSES)]
    empty = sample[sample.label == EMPTY]
    cars = sample[sample.label.isin(VEHICLE_CLASSES)]
    a_conf = animal.file_name.map(lambda f: max_conf(f, "animal"))
    any_conf = empty.file_name.map(lambda f: max((c for _, c, _ in dets[f]), default=0.0))
    v_conf = cars.file_name.map(lambda f: max_conf(f, "vehicle"))
    rows = []
    for t in THRESHOLDS:
        rows.append({
            "threshold": t,
            f"animal images kept (n={len(animal)})": f"{(a_conf >= t).mean():.1%}",
            f"empty images dropped (n={len(empty)})": f"{(any_conf < t).mean():.1%}",
            f"car images found as vehicle (n={len(cars)})": f"{(v_conf >= t).mean():.1%}",
        })
    return pd.DataFrame(rows).set_index("threshold")


def per_class_recall(sample: pd.DataFrame, dets: dict, threshold: float = BOX_THRESHOLD) -> pd.DataFrame:
    """Share of images per label with at least one animal detection >= threshold."""
    found = sample.file_name.map(lambda f: any(c >= threshold for cat, c, _ in dets[f] if cat == "animal"))
    return (pd.DataFrame({"label": sample.label, "found": found})
            .groupby("label").found.agg(images="size", with_animal_detection="mean")
            .assign(with_animal_detection=lambda d: d.with_animal_detection.map("{:.1%}".format)))


def misses_by_box(sample: pd.DataFrame, dets: dict, gt: dict, threshold: float = BOX_THRESHOLD) -> pd.DataFrame:
    """Animal images with/without an annotated box vs. found/missed by the detector.

    CCT labels are given per sequence, so a frame after the animal left still
    carries its label but no box. Misses on such frames are mostly not misses.
    """
    animal = sample[~sample.label.isin({EMPTY} | VEHICLE_CLASSES)]
    found = animal.file_name.map(lambda f: any(c >= threshold for cat, c, _ in dets[f] if cat == "animal"))
    has_box = animal.image_id.map(lambda i: bool(gt.get(i, {}).get("boxes")))
    table = pd.crosstab(has_box.map({True: "annotated box", False: "no box"}),
                        found.map({True: "found", False: "missed"}), margins=True, margins_name="all")
    return table.rename_axis(index="ground truth", columns="MegaDetector")


def box_level(sample: pd.DataFrame, dets: dict, gt: dict, threshold: float = BOX_THRESHOLD) -> tuple[dict, pd.DataFrame]:
    """Box precision, recall and IoU on images whose animals all have real boxes."""
    rows = []
    for r in sample.itertuples():
        g = gt.get(r.image_id)
        if g is None or not g["complete"] or not g["boxes"]:
            continue
        d = [(c, box) for cat, c, box in dets[r.file_name] if cat == "animal" and c >= threshold]
        ious, fp, fn = match(g["boxes"], d)
        rows.append({"label": r.label, "gt_boxes": len(g["boxes"]), "tp": len(ious), "fp": fp, "fn": fn,
                     "iou_sum": sum(ious)})
    df = pd.DataFrame(rows)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    summary = {
        "images": len(df),
        "gt boxes": int(df.gt_boxes.sum()),
        "detections": int(tp + fp),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "mean IoU of matches": df.iou_sum.sum() / tp if tp else float("nan"),
    }
    per_label = df.groupby("label")[["gt_boxes", "tp", "fp", "fn"]].sum()
    per_label["recall"] = (per_label.tp / per_label.gt_boxes).map("{:.1%}".format)
    return summary, per_label


def compare_runs(ours: dict, ref: dict, threshold: float = BOX_THRESHOLD) -> dict:
    """Agreement between two detection runs on the files of ``ours``."""
    agree, diffs, n_ours, n_ref, ious = 0, [], 0, 0, []
    for f, d in ours.items():
        r = ref[f]
        mo = max((c for cat, c, _ in d if cat == "animal"), default=0.0)
        mr = max((c for cat, c, _ in r if cat == "animal"), default=0.0)
        agree += (mo >= threshold) == (mr >= threshold)
        diffs.append(abs(mo - mr))
        a = [box for _, c, box in d if c >= threshold]
        b = [(c, box) for _, c, box in r if c >= threshold]
        matched, _, _ = match(a, b)
        n_ours, n_ref, ious = n_ours + len(a), n_ref + len(b), ious + matched
    return {
        "images": len(ours),
        "same animal / no animal decision": agree / len(ours),
        "mean abs. difference of max animal confidence": sum(diffs) / len(diffs),
        "boxes (this run)": n_ours,
        "boxes (reference)": n_ref,
        "matched boxes (IoU >= 0.5)": len(ious),
        "mean IoU of matched boxes": sum(ious) / len(ious) if ious else float("nan"),
    }


def build_report(sample: pd.DataFrame, dets: dict, gt: dict, model: str, ref: dict | None = None) -> str:
    summary, per_label = box_level(sample, dets, gt)
    lines = [
        "# MegaDetector on a CCT20 sample", "",
        "Generated by `python -m mammalcam.detect.evaluate`. Do not edit by hand.", "",
        f"- Model: MegaDetector {model}, pretrained, not fine-tuned",
        f"- Sample: {len(sample)} images, class-balanced, at most one image per sequence, "
        f"from {sample.location.nunique()} cameras",
        "- Caveat: MegaDetector's training data includes Caltech Camera Traps, so these numbers are "
        "likely optimistic compared to cameras it has never seen.", "",
        "## Image level: empty filter", "",
        "An image counts as kept if it has an animal detection at or above the threshold. "
        "An empty image counts as dropped if it has no detection of any kind at or above the threshold.", "",
        image_level(sample, dets).to_markdown(), "",
        f"### Images with an animal detection (threshold {BOX_THRESHOLD}) per label", "",
        per_class_recall(sample, dets).to_markdown(), "",
        f"### Animal images: missed vs. annotated box (threshold {BOX_THRESHOLD})", "",
        "Labels are given per sequence, so a frame after the animal has left still carries its label. "
        "Such frames usually have no box.", "",
        misses_by_box(sample, dets, gt).to_markdown(), "",
        f"## Box level (threshold {BOX_THRESHOLD}, IoU ≥ {IOU_THRESHOLD})", "",
        "Only images whose animals all have a real box (no missing or full-frame boxes); "
        "empty and car images are left out.", "",
    ]
    lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in summary.items()]
    lines += ["", per_label.to_markdown(), ""]
    if ref is not None:
        lines += [f"## Comparison with LILA's precomputed MegaDetector v5a results (threshold {BOX_THRESHOLD})", "",
                  "LILA ran the detector on the full-resolution originals, this run used the 1024 px download. "
                  "The LILA results are used to crop all images.", ""]
        lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
                  for k, v in compare_runs(dets, ref).items()]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--detections", type=Path, default=Path("data/detections/md_sample.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/02_megadetector_sample.md"))
    parser.add_argument("--reference", type=Path, default=LILA_RESULTS, help="other run to compare with (skipped if missing)")
    args = parser.parse_args(argv)

    images, ann = load_cct(args.ann_dir)
    files = pd.read_csv(args.detections.with_suffix(".csv")).file_name
    sample = images[images.file_name.isin(files)]
    dets = load_detections(args.detections)
    missing = set(sample.file_name) - set(dets)
    if missing:
        raise RuntimeError(f"{len(missing)} sample images have no detection result")
    model = json.loads(args.detections.read_text()).get("info", {}).get("detector", "unknown")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gt = ground_truth(images, ann[ann.image_id.isin(sample.image_id)])
    ref = load_detections(args.reference) if args.reference.exists() else None
    args.out.write_text(build_report(sample, dets, gt, Path(model).stem, ref))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
