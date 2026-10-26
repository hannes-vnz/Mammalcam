"""Example figures for the README: real test images with boxes and predictions.

Usage:
    python -m mammalcam.eval.examples [--seed 0]

Every prediction shown comes from the camera cross-validation: the model that
labels an image never saw that camera. Boxes are MegaDetector's (conf >= 0.2);
each box shows the crop classifier's class and probability, green if it
matches the annotated label, red if not. Images with a person detection are
skipped.

Needs data/predictions/classify_test.csv (python -m mammalcam.models.classify).
Writes docs/examples/*.jpg and docs/examples/sources.csv (image, camera,
rights holder per tile, for attribution).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mammalcam.data.cct import DEFAULT_ANN_DIR, DEFAULT_IMG_DIR, EMPTY, load_cct
from mammalcam.data.split import fold_parts
from mammalcam.detect.evaluate import LILA_RESULTS, load_detections
from mammalcam.models.classify import NO_DETECTION, Data

OUT = Path("docs/examples")
TILE_W = 480
GREEN, RED, GREY, YELLOW = (46, 160, 67), (215, 58, 73), (90, 90, 90), (230, 180, 0)


def crop_predictions(data: Data, folds: dict, cv_c: list[float]) -> pd.DataFrame:
    """Class and probability for every detection, from the fold model that did not see its camera."""
    k = len(cv_c)
    out = []
    loc_of = data.images.set_index("image_id").location
    for f in range(k):
        part = fold_parts(folds, f, k)
        tc = data.train_crops
        train = tc[tc.image_id.map(loc_of).map(part).values == "train"]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=cv_c[f], class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(data.x_dets[train.row.values], train.label.values)
        d = data.dets[data.dets.location.map(part) == "test"]
        proba = model.predict_proba(data.x_dets[d.index.values])
        out.append(d[["crop_file", "image_id", "conf", "bbox"]].assign(
            pred=model.classes_[proba.argmax(1)], prob=proba.max(1)))
        print(f"fold {f}: {len(d)} crops")
    return pd.concat(out)


def font(size: int):
    return ImageFont.load_default(size=size)


def _label(draw, box, text, colour, placed: list) -> None:
    """Text tag above the box, or below it if that would overlap an earlier tag."""
    small = font(15)
    tw = draw.textlength(text, font=small) + 8
    x = min(box[0], draw.im.size[0] - tw)  # keep the tag inside the image
    for ty in (max(box[1] - 20, 0), box[3], box[3] + 22):
        rect = (x, ty, x + tw, ty + 20)
        if not any(rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3] for r in placed):
            break
    placed.append(rect)
    draw.rectangle(rect, fill=colour)
    draw.text((rect[0] + 4, rect[1] + 2), text, fill="white", font=small)


def render(row, crops: pd.DataFrame, img_dir: Path, header: list[tuple[str, tuple]], width: int = TILE_W,
           gt_boxes: list | None = None):
    """One tile: image with labelled boxes and a header bar of (text, colour) parts.

    gt_boxes: optional annotated boxes (relative x, y, w, h) drawn in yellow.
    """
    im = Image.open(img_dir / row.file_name).convert("RGB")
    scale = width / im.width
    im = im.resize((width, round(im.height * scale)))
    draw = ImageDraw.Draw(im)
    placed = []

    def px(b):
        x, y, w, h = b
        return [x * im.width, y * im.height, (x + w) * im.width, (y + h) * im.height]

    for b in gt_boxes or []:
        box = px(b)
        draw.rectangle(box, outline=YELLOW, width=3)
        _label(draw, box, f"annotated: {row.label}", YELLOW, placed)
    for c in crops.itertuples():
        box = px([float(v) for v in c.bbox.split()])
        colour = GREEN if c.pred == row.label else RED
        draw.rectangle(box, outline=colour, width=3)
        _label(draw, box, f"{c.pred} {c.prob:.0%}", colour, placed)
    bar = Image.new("RGB", (width, 30), (245, 245, 245))
    d = ImageDraw.Draw(bar)
    x = 8
    for text, colour in header:
        d.text((x, 6), text, fill=colour, font=font(16))
        x += d.textlength(text, font=font(16))
    tile = Image.new("RGB", (width, im.height + bar.height), "white")
    tile.paste(bar, (0, 0))
    tile.paste(im, (0, bar.height))
    return tile


def grid(tiles: list[Image.Image], cols: int, path: Path, gap: int = 6) -> None:
    rows = -(-len(tiles) // cols)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * TILE_W + (cols - 1) * gap, rows * h + (rows - 1) * gap), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (TILE_W + gap), (i // cols) * (h + gap)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=82, optimize=True)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


def verdict(pred: str, true: str) -> tuple[str, tuple]:
    if pred == NO_DETECTION:
        return "no detection", RED
    return pred, GREEN if pred == true else RED


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    res = json.loads(Path("reports/results/classify.json").read_text())
    folds = {int(loc): int(f) for f, locs in res["cv"]["camera_groups"].items() for loc in locs}
    data = Data(args.ann_dir, args.img_dir, workers=4)
    cache = Path("data/predictions/crops_cv.csv")
    if cache.exists():
        crops = pd.read_csv(cache)
    else:
        crops = crop_predictions(data, folds, [f["C"]["crops"] for f in res["cv"]["folds"]])
        crops.to_csv(cache, index=False)
    by_image = {i: g for i, g in crops.groupby("image_id")}
    empty = crops.iloc[:0]

    preds = pd.read_csv("data/predictions/classify_test.csv")
    preds = preds[preds.evaluation.str.startswith("cv_fold")]
    p = preds.merge(data.images[["image_id", "file_name", "frame_num", "n_boxes", "width", "height"]], on="image_id")
    rights = {im["file_name"]: im.get("rights_holder", "")
              for f in Path(args.ann_dir).glob("*_annotations.json") for im in json.loads(f.read_text())["images"]}
    sources = []

    def tile(r, header, figure, gt=None):
        sources.append({"figure": figure, "file_name": r.file_name, "camera": r.location, "label": r.label,
                        "rights_holder": rights.get(r.file_name, "")})
        return render(r, by_image.get(r.image_id, empty), args.img_dir, header, gt_boxes=gt)
    _, ann = load_cct(args.ann_dir)
    ann = ann[(ann.category != EMPTY) & ann.bbox.notna()]
    p["n_crops"] = p.image_id.map(lambda i: len(by_image[i]) if i in by_image else 0)
    md = load_detections(LILA_RESULTS)
    person = p.file_name.map(lambda f: any(cat == "person" and c >= 0.1 for cat, c, _ in md[f]))
    p = p[~person]
    top = p.image_id.map(lambda i: by_image[i].prob.max() if i in by_image else 0.0)

    def pick(df, n=1):
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        return [df.iloc[i] for i in idx]

    def header_for(r):
        return [(f"true: {r.label}  |  model: ", GREY), verdict(r.crops_image, r.label),
                (f"  |  camera {r.location}", GREY)]

    # 1. correct predictions, one per species, confident and on different cameras
    tiles, used = [], set()
    for cls in ["coyote", "bobcat", "raccoon", "skunk", "rabbit", "squirrel"]:
        cand = p[(p.label == cls) & (p.crops_image == cls) & (top > 0.8) & ~p.location.isin(used)]
        r = pick(cand)[0]
        used.add(r.location)
        tiles.append(tile(r, header_for(r), "correct"))
    grid(tiles, 3, OUT / "correct.jpg")

    # 2. background trap: the full-image model is wrong, the crop model right
    tiles = []
    for cls in ["rodent", "cat", "bobcat"]:
        cand = p[(p.label == cls) & (p.crops_image == cls) & (p.full_image != cls) & (p.n_crops == 1)]
        r = pick(cand)[0]
        header = [(f"true: {r.label}  |  full image: ", GREY), verdict(r.full_image, r.label),
                  ("  |  crop: ", GREY), verdict(r.crops_image, r.label)]
        tiles.append(tile(r, header, "full_vs_crops"))
    grid(tiles, 3, OUT / "full_vs_crops.jpg")

    # 3. a burst where averaging fixes single frames
    seq = p.groupby("seq_id").filter(
        lambda g: len(g) == 3 and (g.crops_sequence == g.label).all() and (g.crops_image != g.label).sum() >= 2
        and (g.crops_image == g.label).any())
    s = seq[seq.seq_id == pick(seq.drop_duplicates("seq_id"))[0].seq_id].sort_values("frame_num")
    tiles = []
    for r in s.itertuples():
        header = [(f"frame {r.frame_num}/3: ", GREY), verdict(r.crops_image, r.label),
                  ("  |  burst: ", GREY), verdict(r.crops_sequence, r.label)]
        tiles.append(tile(r, header, "burst"))
    grid(tiles, 3, OUT / "burst.jpg")

    # 4. failures: wrong on the frame and still wrong after the burst average;
    #    for the missed detection, the annotated box shows the animal that was there
    tiles = []
    wrong = (p.crops_image != p.label) & (p.crops_sequence != p.label)
    for cond, show_gt in [((p.label == "deer") & (p.crops_image == "coyote"), False),
                          ((p.label == "rodent") & ~p.crops_image.isin(["rodent", NO_DETECTION]), False),
                          ((p.crops_image == NO_DETECTION) & (p.n_boxes == 1), True)]:
        r = pick(p[cond & wrong])[0]
        header = [(f"true: {r.label}  |  model: ", GREY), verdict(r.crops_image, r.label),
                  (f"  |  camera {r.location}", GREY)]
        gt = None
        if show_gt:
            gt = [(b[0] / r.width, b[1] / r.height, b[2] / r.width, b[3] / r.height)
                  for b in ann[ann.image_id == r.image_id].bbox]
        tiles.append(tile(r, header, "failures", gt))
    grid(tiles, 3, OUT / "failures.jpg")
    pd.DataFrame(sources).to_csv(OUT / "sources.csv", index=False)


if __name__ == "__main__":
    main()
