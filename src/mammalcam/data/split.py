"""Train/val/test splits: by camera location, and random splits for comparison.

The location split is the one used for all reported results: no camera
appears in more than one part, so the test set measures performance on
unseen cameras. The random splits have the same sizes and exist only to
show how much a careless split overestimates accuracy:

- ``random_image``: each image assigned independently. Frames of one burst
  (sequence) end up in train and test, which is close to label leakage.
- ``random_sequence``: whole sequences assigned. No burst leakage, but the
  same cameras and backgrounds appear in train and test.

Usage:
    python -m mammalcam.data.split [--seed 0]

Writes the chosen locations to configs/splits_cct20.json (tracked) and one
CSV per strategy to data/splits/ (not tracked).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mammalcam.data.cct import DEFAULT_ANN_DIR, MULTI, load_cct

SPLITS = ("train", "val", "test")


def _assign_locations(sizes: pd.Series, order: list, val_frac: float, test_frac: float) -> dict:
    """Fill test, then val, in the given location order; the rest is train."""
    total = sizes.sum()
    out, test_n, val_n = {}, 0, 0
    for loc in order:
        if test_n < test_frac * total:
            out[loc] = "test"
            test_n += sizes[loc]
        elif val_n < val_frac * total:
            out[loc] = "val"
            val_n += sizes[loc]
        else:
            out[loc] = "train"
    return out


def _score(assign: dict, loc_labels: dict, needed: set, sizes: pd.Series, val_frac: float, test_frac: float) -> float:
    """Lower is better: missing classes weigh far more than size deviations."""
    present = {s: set() for s in SPLITS}
    count = dict.fromkeys(SPLITS, 0)
    for loc, s in assign.items():
        present[s] |= loc_labels[loc]
        count[s] += sizes[loc]
    missing = len(needed - present["train"]) + len(needed - present["test"])
    missing_val = len(needed - present["val"])
    total = sizes.sum()
    dev = abs(count["test"] / total - test_frac) + abs(count["val"] / total - val_frac)
    return 100 * missing + 10 * missing_val + dev


def location_split(
    images: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.25,
    seed: int = 0,
    n_trials: int = 2000,
) -> dict:
    """Return {location: split}. Whole locations only.

    Tries ``n_trials`` random location orders and keeps the one where the
    most classes occur in train and test and the sizes are closest to the
    targets. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    sizes = images.groupby("location").size()
    locs = list(sizes.index)
    single = images[images.label != MULTI]
    loc_labels = {loc: set(single.label[single.location == loc]) for loc in locs}
    # a class can only be in train and test if it occurs at two or more locations
    n_locs = single.groupby("label").location.nunique()
    needed = set(n_locs[n_locs >= 2].index)
    best, best_score = None, np.inf
    for _ in range(n_trials):
        order = list(rng.permutation(locs))
        assign = _assign_locations(sizes, order, val_frac, test_frac)
        if not set(SPLITS) <= set(assign.values()):
            continue
        score = _score(assign, loc_labels, needed, sizes, val_frac, test_frac)
        if score < best_score:
            best, best_score = assign, score
    if best is None:
        raise ValueError("could not build a split with train, val and test locations")
    return {int(k): v for k, v in sorted(best.items())}


def camera_folds(images: pd.DataFrame, k: int = 5, seed: int = 0, n_trials: int = 2000) -> dict:
    """Assign every camera to one of k groups for cross-validation: {location: fold}.

    Groups get similar numbers of images, and in every round of
    ``fold_parts`` (one group test, the next one validation) each class seen
    at two or more cameras still has a camera in the training groups.
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    sizes = images.groupby("location").size()
    single = images[images.label != MULTI]
    class_locs = single.groupby("label").location.agg(set)
    class_locs = class_locs[class_locs.map(len) >= 2]
    best, best_score = None, np.inf
    for _ in range(n_trials):
        load = np.zeros(k)
        assign = {}
        for loc in rng.permutation(sizes.index):  # greedy: next camera to the smallest group
            fold = int(np.argmin(load))
            assign[int(loc)] = fold
            load[fold] += sizes[loc]
        stuck = sum(
            {assign[loc] for loc in locs} <= {f, (f + 1) % k}  # only in this round's test and val groups
            for f in range(k) for locs in class_locs
        )
        score = 100 * stuck + (load.max() - load.min()) / load.sum()
        if score < best_score:
            best, best_score = assign, score
    return dict(sorted(best.items()))


def fold_parts(folds: dict, test_fold: int, k: int) -> dict:
    """{location: part} for one cross-validation round: test group, the next group as val, rest train."""
    val_fold = (test_fold + 1) % k
    return {loc: "test" if f == test_fold else "val" if f == val_fold else "train" for loc, f in folds.items()}


def random_split(images: pd.DataFrame, counts: dict, by: str = "image", seed: int = 0) -> pd.Series:
    """Random split with (about) the given number of images per part.

    by="image" shuffles images, by="sequence" shuffles whole sequences.
    Returns a Series of split names aligned with ``images.index``.
    """
    rng = np.random.default_rng(seed)
    if by == "image":
        order = rng.permutation(len(images))
        names = np.repeat(SPLITS, [counts.get(s, 0) for s in SPLITS])
        if len(names) != len(images):
            raise ValueError("counts must add up to the number of images")
        out = np.empty(len(images), dtype=object)
        out[order] = names
        return pd.Series(out, index=images.index, name="split")
    if by == "sequence":
        seq_sizes = images.groupby("seq_id").size()
        seqs = seq_sizes.index[rng.permutation(len(seq_sizes))]
        cum = seq_sizes[seqs].cumsum()
        test_end, val_end = counts["test"], counts["test"] + counts["val"]
        seq_split = pd.Series(
            np.where(cum <= test_end, "test", np.where(cum <= val_end, "val", "train")), index=seqs
        )
        return images.seq_id.map(seq_split).rename("split")
    raise ValueError(f"unknown by={by!r}")


def overlap(images: pd.DataFrame, split: pd.Series, key: str) -> dict:
    """Values of ``key`` (e.g. location, seq_id) that appear in more than one split."""
    groups = pd.DataFrame({"key": images[key], "split": split}).groupby("key").split.nunique()
    return {"n_shared": int((groups > 1).sum()), "n_total": int(len(groups))}


def build_all(images: pd.DataFrame, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.25):
    """Return (location assignment, {strategy: split Series})."""
    assign = location_split(images, val_frac, test_frac, seed)
    loc = images.location.map(assign).rename("split")
    counts = loc.value_counts().to_dict()
    return assign, {
        "location": loc,
        "random_image": random_split(images, counts, by="image", seed=seed),
        "random_sequence": random_split(images, counts, by="sequence", seed=seed),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=Path, default=Path("configs/splits_cct20.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args(argv)

    images, _ = load_cct(args.ann_dir)
    assign, splits = build_all(images, args.seed, args.val_frac, args.test_frac)

    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps({
        "dataset": "cct20",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "locations": {s: [k for k, v in assign.items() if v == s] for s in SPLITS},
    }, indent=2) + "\n")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        pd.DataFrame({"image_id": images.image_id, "split": split}).to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name:16s}", split.value_counts().reindex(SPLITS).to_dict(),
              "shared locations:", overlap(images, split, "location")["n_shared"],
              "shared sequences:", overlap(images, split, "seq_id")["n_shared"])
    print(f"locations -> {args.config}, splits -> {args.out}/")


if __name__ == "__main__":
    main()
