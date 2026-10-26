"""Download datasets from LILA BC into data/raw.

Usage:
    python -m mammalcam.data.download {cct20,cct-md} [--root data/raw] [--keep-archives]

Downloads resume after an interruption, are checked against the MD5 that
Google Cloud Storage reports (where it reports one), and are extracted with
tarfile's "data" filter or zipfile, which both refuse absolute paths and
paths outside the target directory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import sys
import tarfile
import zipfile
import urllib.request
from pathlib import Path

GCS = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
LILA_MD = "https://lila.science/public/lila-md-results"

# CCT20 benchmark subset of Caltech Camera Traps (Beery et al. 2018,
# "Recognition in Terra Incognita"). Licence: CDLA-Permissive 1.0.
DATASETS = {
    "cct20": {
        "dir": "cct20",
        "files": [
            f"{GCS}/eccv_18_annotations.tar.gz",
            f"{GCS}/eccv_18_all_images_sm.tar.gz",
        ],
    },
    # MegaDetector v5a results for all of Caltech Camera Traps, precomputed by
    # LILA BC. Used to crop animals, not to evaluate MegaDetector (it was
    # trained on this data).
    "cct-md": {
        "dir": "cct20/megadetector",
        "files": [f"{LILA_MD}/caltech-camera-traps_mdv5a.0.0_results.json.zip"],
    },
}

CHUNK = 1 << 20


def remote_info(url: str) -> tuple[int, str | None]:
    """Return (size in bytes, base64 MD5 or None) from a HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        size = int(resp.headers["Content-Length"])
        md5 = None
        for value in resp.headers.get_all("x-goog-hash") or []:
            for part in value.split(","):
                key, _, digest = part.strip().partition("=")
                if key == "md5":
                    md5 = digest
        return size, md5


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def download(url: str, dest: Path) -> Path:
    """Download url to dest, resuming a partial file, and verify its MD5."""
    size, md5 = remote_info(url)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == size:
        print(f"already there: {dest.name}")
    else:
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req) as resp, part.open("ab" if have else "wb") as out:
            if have and resp.status != 206:  # server ignored the range: start over
                out.seek(0)
                out.truncate()
                have = 0
            done = have
            for chunk in iter(lambda: resp.read(CHUNK), b""):
                out.write(chunk)
                done += len(chunk)
                print(f"\r{dest.name}: {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
        print()
        if part.stat().st_size != size:
            sys.exit(f"incomplete download: {dest.name}, run again to resume")
        part.rename(dest)
    if md5 and md5_b64(dest) != md5:
        dest.unlink()
        sys.exit(f"MD5 mismatch, file removed: {dest.name}")
    return dest


def extract(archive: Path, target: Path) -> None:
    print(f"extracting {archive.name} ...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--keep-archives", action="store_true", help="keep the archives after extraction")
    args = parser.parse_args(argv)

    spec = DATASETS[args.dataset]
    target = args.root / spec["dir"]
    archives = target / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    sizes = [remote_info(url)[0] for url in spec["files"]]
    free = shutil.disk_usage(target).free
    print(f"{args.dataset}: {len(sizes)} files, {sum(sizes) / 1e9:.1f} GB to download, {free / 1e9:.0f} GB free")

    for url in spec["files"]:
        name = url.rsplit("/", 1)[1]
        marker = target / f".extracted-{name}"
        if marker.exists():
            print(f"already extracted: {name}")
            continue
        archive = download(url, archives / name)
        extract(archive, target)
        marker.touch()
        if not args.keep_archives:
            archive.unlink()
    print(f"done: {target}")


if __name__ == "__main__":
    main()
