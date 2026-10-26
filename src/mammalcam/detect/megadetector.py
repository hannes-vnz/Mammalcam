"""Run MegaDetector (pretrained, inference only) on CCT20 images.

Usage:
    python -m mammalcam.detect.megadetector --sample 500 [--seed 0]

Downloads the model weights once into models/ (MD5-checked), draws a
class-balanced sample (one image per sequence), runs the detector on MPS,
CUDA or CPU, and writes the results in MegaDetector's batch JSON format to
data/detections/. Boxes in that format are [x, y, w, h], relative to the
image size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from mammalcam.data.cct import DEFAULT_ANN_DIR, DEFAULT_IMG_DIR, load_cct
from mammalcam.data.sample import stratified_sample

MODEL_VERSION = "v5a.0.1"
# MegaDetector's own category ids in its output JSON
MD_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}


def model_path(models_dir: Path = Path("models"), version: str = MODEL_VERSION) -> Path:
    """Local path of the weights; downloads and MD5-checks them on first use."""
    from megadetector.detection.run_detector import known_models

    info = known_models[version]
    dest = models_dir / info["url"].rsplit("/", 1)[1]
    if not dest.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        print(f"downloading {info['url']}")
        urllib.request.urlretrieve(info["url"], part)
        part.rename(dest)
    digest = hashlib.md5(dest.read_bytes()).hexdigest()
    if digest != info["md5"]:
        dest.unlink()
        raise RuntimeError(f"MD5 mismatch for {dest.name}, file removed")
    return dest


def run(image_files: list[Path], out: Path, threshold: float = 0.01) -> dict:
    """Run the detector on image_files and write the batch JSON to out.

    The low threshold keeps weak detections so the evaluation can sweep
    thresholds afterwards.
    """
    from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file

    weights = str(model_path())
    results = load_and_run_detector_batch(
        weights, [str(p) for p in image_files], confidence_threshold=threshold, quiet=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_results_to_file(results, str(out), detector_file=weights)
    return json.loads(out.read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/detections/md_sample.json"))
    args = parser.parse_args(argv)

    images, _ = load_cct(args.ann_dir)
    sample = stratified_sample(images, args.sample, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample[["image_id", "file_name", "label", "location"]].to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"{len(sample)} images, {sample.label.nunique()} labels -> running MegaDetector {MODEL_VERSION}")
    run([args.img_dir / f for f in sample.file_name], args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
