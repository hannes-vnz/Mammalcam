"""Prepare SpeciesNet input from the CCT20 test images and LILA's MegaDetector boxes.

Usage (main environment):
    python -m mammalcam.speciesnet.prepare [--limit N]

Writes, in data/speciesnet/:
- instances.json: the test images (same as for the own classifiers), with
  country USA and state CA so SpeciesNet's geofence applies
- detections.json: LILA's MegaDetector v5a boxes in SpeciesNet's format, so
  SpeciesNet does not have to run the detector again

SpeciesNet itself runs in .venv-speciesnet (see run.sh in this folder).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mammalcam.data.cct import DEFAULT_ANN_DIR, DEFAULT_IMG_DIR, load_cct
from mammalcam.detect.evaluate import LILA_RESULTS
from mammalcam.models.baseline import load_classes, select_images

OUT = Path("data/speciesnet")
CATEGORY = {"1": "animal", "2": "human", "3": "vehicle"}  # SpeciesNet calls MegaDetector's "person" "human"


def to_speciesnet_detections(md_results: dict, filepaths: dict[str, str]) -> dict:
    """MegaDetector batch JSON -> SpeciesNet detections JSON for the given files.

    filepaths: file name -> full path as used in instances.json.
    """
    preds = []
    for im in md_results["images"]:
        path = filepaths.get(Path(im["file"]).name)
        if path is None:
            continue
        dets = [{"category": d["category"], "label": CATEGORY[d["category"]], "conf": d["conf"], "bbox": d["bbox"]}
                for d in im["detections"]]
        preds.append({"filepath": path, "detections": sorted(dets, key=lambda d: -d["conf"])})
    return {"predictions": preds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (speed test)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    images, _ = load_cct(args.ann_dir)
    table = select_images(images, load_classes())
    if args.limit:
        table = table.sample(args.limit, random_state=0)
    filepaths = {f: str((args.img_dir / f).resolve()) for f in table.file_name}

    args.out.mkdir(parents=True, exist_ok=True)
    instances = [{"filepath": p, "country": "USA", "admin1_region": "CA"} for p in filepaths.values()]
    (args.out / "instances.json").write_text(json.dumps({"instances": instances}))
    md = json.loads(LILA_RESULTS.read_text())
    dets = to_speciesnet_detections(md, filepaths)
    if len(dets["predictions"]) != len(filepaths):
        raise RuntimeError("some test images have no MegaDetector entry")
    (args.out / "detections.json").write_text(json.dumps(dets))
    print(f"{len(instances)} instances -> {args.out}")


if __name__ == "__main__":
    main()
