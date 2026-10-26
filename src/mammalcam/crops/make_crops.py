"""Cut animal crops out of the CCT20 images for the species classifier.

Usage:
    python -m mammalcam.crops.make_crops [--threshold 0.2] [--size 256] [--workers 8]

Boxes come from MegaDetector (LILA's precomputed v5a results). Each animal
detection at or above the threshold becomes one crop, labelled like this:

- image with one species: the detection gets that species. If the image
  also has complete ground-truth boxes, only detections that overlap one of
  them (IoU >= 0.5) are kept, which removes false positives on rocks or
  branches. Without ground-truth boxes the image label is used as is
  (source "image_label"), as it would be for any detector output.
- image with several species: the species of the matching ground-truth box;
  detections without a match are dropped.
- empty and car images: no crops (vehicles are handled by stage 1).

Crops are square, padded around the box, shifted inside the image where
possible and resized so the longer side is at most ``--size`` pixels.
Writes data/crops/images/*.jpg and data/crops/crops.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from mammalcam.data.cct import DEFAULT_ANN_DIR, DEFAULT_IMG_DIR, EMPTY, MULTI, load_cct
from mammalcam.detect.evaluate import IOU_THRESHOLD, LILA_RESULTS, VEHICLE_CLASSES, iou, load_detections, to_relative

NO_CROP_LABELS = {EMPTY} | VEHICLE_CLASSES
DOWNLOAD_MAX_SIDE = 1024  # CCT20 images are downscaled to this longer side
MIN_SIDE_PX = 12  # detections smaller than this (in the downloaded image) are dropped


def square_box(bbox, width: int, height: int, pad: float = 0.1) -> tuple[int, int, int, int]:
    """Relative [x, y, w, h] -> pixel (left, top, right, bottom) of a padded square.

    The square is shifted to stay inside the image; if it is larger than the
    image in one direction it is clipped there.
    """
    x, y, w, h = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = x + w / 2, y + h / 2

    def place(center, size):
        lo = center - side / 2
        lo = min(max(lo, 0), max(size - side, 0))
        return int(round(lo)), int(round(min(lo + side, size)))

    left, right = place(cx, width)
    top, bottom = place(cy, height)
    return left, top, right, bottom


def label_detections(row, ann_boxes: list[tuple[str, tuple]], complete: bool, dets: list,
                     threshold: float = 0.2, iou_threshold: float = IOU_THRESHOLD) -> list[dict]:
    """Labelled crops for one image.

    row: image record with ``label``, ``width``, ``height``.
    ann_boxes: [(category, relative box)] ground-truth animal boxes.
    complete: True if every animal on the image has a real box.
    dets: [(category, confidence, relative box)] detector output.
    """
    if row.label in NO_CROP_LABELS:
        return []
    out = []
    free = list(range(len(ann_boxes)))
    animals = sorted([(c, b) for cat, c, b in dets if cat == "animal" and c >= threshold], key=lambda d: -d[0])
    for conf, box in animals:
        best_iou, best = max(((iou(box, ann_boxes[i][1]), i) for i in free), default=(0.0, None))
        if best is not None and best_iou >= iou_threshold:
            free.remove(best)
            out.append({"label": ann_boxes[best][0], "conf": conf, "bbox": box, "source": "box_match"})
        elif row.label != MULTI and not (complete and ann_boxes):
            out.append({"label": row.label, "conf": conf, "bbox": box, "source": "image_label"})
    return out


def _crop_one(job: tuple) -> list[str]:
    """Worker: cut all crops of one image. Returns the crop file names written."""
    src, crops, out_dir, size = job
    written = []
    with Image.open(src) as im:
        im = im.convert("RGB")
        width, height = im.size
        for name, bbox in crops:
            box = square_box(bbox, width, height)
            crop = im.crop(box)
            crop.thumbnail((size, size))
            crop.save(out_dir / name, quality=90)
            written.append(name)
    return written


def build_crop_table(images: pd.DataFrame, ann: pd.DataFrame, dets: dict, threshold: float) -> pd.DataFrame:
    """One row per crop with label, source, box and image metadata (no pixels touched)."""
    meta = images.set_index("image_id")
    animal_ann = ann[(ann.category != EMPTY)]
    boxes, complete = {}, {}
    for image_id, g in animal_ann.groupby("image_id"):
        w, h = meta.loc[image_id, ["width", "height"]]
        rel = [(cat, to_relative(b, w, h)) for cat, b in zip(g.category, g.bbox) if b is not None]
        rel = [(cat, b) for cat, b in rel if b[2] * b[3] <= 0.9]  # full-frame boxes are not real boxes
        boxes[image_id] = rel
        complete[image_id] = len(rel) == len(g)

    rows = []
    for r in images.itertuples():
        for k, c in enumerate(label_detections(r, boxes.get(r.image_id, []), complete.get(r.image_id, False),
                                               dets.get(r.file_name, []), threshold)):
            rows.append({"crop_file": f"{r.image_id}_{k}.jpg", "image_id": r.image_id, "file_name": r.file_name,
                         "location": r.location, "seq_id": r.seq_id, "image_label": r.label, **c})
    crops = pd.DataFrame(rows)
    # tiny detections are not usable as classifier input; size measured in the downloaded image
    w = crops.image_id.map(meta.width).values
    h = crops.image_id.map(meta.height).values
    scale = (DOWNLOAD_MAX_SIDE / pd.Series(w).combine(pd.Series(h), max)).clip(upper=1).values
    px = crops.bbox.map(lambda b: b[2]).values * w * scale
    py = crops.bbox.map(lambda b: b[3]).values * h * scale
    return crops[(px >= MIN_SIDE_PX) & (py >= MIN_SIDE_PX)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--detections", type=Path, default=LILA_RESULTS)
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("data/crops"))
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    images, ann = load_cct(args.ann_dir)
    crops = build_crop_table(images, ann, load_detections(args.detections), args.threshold)
    for name in ("location", "random_sequence", "random_image"):
        split = pd.read_csv(args.splits / f"{name}.csv").set_index("image_id").split
        crops[f"split_{name}"] = crops.image_id.map(split)

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.img_dir / f, list(zip(g.crop_file, g.bbox)), img_dir, args.size)
            for f, g in crops.groupby("file_name")]
    with ProcessPoolExecutor(args.workers) as pool:
        n = sum(len(w) for w in pool.map(_crop_one, jobs, chunksize=64))
    crops.drop(columns="file_name").assign(bbox=crops.bbox.map(lambda b: " ".join(f"{v:.4f}" for v in b))) \
        .to_csv(args.out / "crops.csv", index=False)
    print(f"{n} crops from {len(jobs)} images -> {args.out}")


if __name__ == "__main__":
    main()
