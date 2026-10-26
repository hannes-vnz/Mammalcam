"""Load COCO Camera Traps metadata (CCT20) into flat tables.

CCT20 ships as five JSON files (train, cis_val, cis_test, trans_val,
trans_test). They are merged here; the original file name is kept in the
column ``benchmark_split`` so the published benchmark split stays available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EMPTY = "empty"
MULTI = "multi"  # image with more than one animal category

DEFAULT_ANN_DIR = Path("data/raw/cct20/eccv_18_annotation_files")
DEFAULT_IMG_DIR = Path("data/raw/cct20/eccv_18_all_images_sm")


def image_label(categories: list[str]) -> str:
    """One label per image: the single category, or MULTI if there are several.

    ``empty`` only counts when nothing else is annotated on the image.
    """
    cats = set(categories)
    if len(cats) > 1:
        cats.discard(EMPTY)
    if not cats:
        return EMPTY
    return cats.pop() if len(cats) == 1 else MULTI


def load_cct(ann_dir: Path = DEFAULT_ANN_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (images, annotations), one row per image / per annotation.

    images: image_id, file_name, location, seq_id, frame_num, seq_num_frames,
            date_captured, width, height, benchmark_split, label, n_boxes
    annotations: annotation_id, image_id, category, bbox (x, y, w, h in pixels or None)
    """
    files = sorted(Path(ann_dir).glob("*_annotations.json"))
    if not files:
        raise FileNotFoundError(f"no *_annotations.json in {ann_dir}")

    images, annos = [], []
    for f in files:
        data = json.loads(f.read_text())
        cats = {c["id"]: c["name"] for c in data["categories"]}
        split = f.stem.removesuffix("_annotations")
        for im in data["images"]:
            images.append({
                "image_id": im["id"],
                "file_name": im["file_name"],
                "location": int(im["location"]),
                "seq_id": im["seq_id"],
                "frame_num": im["frame_num"],
                "seq_num_frames": im["seq_num_frames"],
                "date_captured": im["date_captured"],
                "width": im["width"],
                "height": im["height"],
                "benchmark_split": split,
            })
        for a in data["annotations"]:
            bbox = a.get("bbox")
            annos.append({
                "annotation_id": a["id"],
                "image_id": a["image_id"],
                "category": cats[a["category_id"]],
                "bbox": tuple(bbox) if bbox else None,
            })

    img = pd.DataFrame(images)
    ann = pd.DataFrame(annos)
    if img.image_id.duplicated().any():
        raise ValueError("image listed in more than one annotation file")
    missing = set(ann.image_id) - set(img.image_id)
    if missing:
        raise ValueError(f"{len(missing)} annotations point to unknown images")

    labels = ann.groupby("image_id")["category"].agg(lambda s: image_label(list(s)))
    boxes = ann.dropna(subset=["bbox"]).groupby("image_id").size()
    img["label"] = img.image_id.map(labels).fillna(EMPTY)
    img["n_boxes"] = img.image_id.map(boxes).fillna(0).astype(int)
    img["date_captured"] = pd.to_datetime(img.date_captured, errors="coerce")
    return img, ann
