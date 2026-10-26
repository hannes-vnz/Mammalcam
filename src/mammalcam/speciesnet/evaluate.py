"""Compare SpeciesNet with the own classifiers on the same test images.

Usage:
    python -m mammalcam.speciesnet.evaluate [--predictions data/speciesnet/predictions.json]

SpeciesNet predicts one of 2,000+ taxonomy labels per detection. The image
prediction is the processed classification (after geofence and roll-up) of
the most confident detection, as for the own crop classifier. It is mapped
to the 12 classes here; a prediction that is correct but coarser than the
class (e.g. "canine family" for a coyote) cannot be mapped and counts as
wrong, but is reported separately.

The own models are taken from the camera cross-validation, so every image
was predicted by a model that never saw its camera.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from mammalcam.data.cct import load_cct
from mammalcam.eval import metrics
from mammalcam.models.baseline import load_classes
from mammalcam.models.classify import NO_DETECTION

# (taxonomy level, value) -> class, checked in this order; levels: class, order, family, genus, species
TAXON_RULES = [
    (("genus", "species"), ("lynx", "rufus"), "bobcat"),
    (("genus", "species"), ("felis", "catus"), "cat"),
    (("genus", "species"), ("canis", "latrans"), "coyote"),
    (("genus", "species"), ("canis", "familiaris"), "dog"),
    (("genus",), ("procyon",), "raccoon"),
    (("family",), ("didelphidae",), "opossum"),
    (("family",), ("leporidae",), "rabbit"),
    (("family",), ("mephitidae",), "skunk"),
    (("family",), ("cervidae",), "deer"),
    (("family",), ("sciuridae",), "squirrel"),
    (("order",), ("rodentia",), "rodent"),
    (("class",), ("aves",), "bird"),
]
LEVELS = ("class", "order", "family", "genus", "species")
# taxonomy each class stands for, down to the level the class covers
REFERENCE = {
    "bobcat": ("mammalia", "carnivora", "felidae", "lynx", "rufus"),
    "cat": ("mammalia", "carnivora", "felidae", "felis", "catus"),
    "coyote": ("mammalia", "carnivora", "canidae", "canis", "latrans"),
    "dog": ("mammalia", "carnivora", "canidae", "canis", "familiaris"),
    "raccoon": ("mammalia", "carnivora", "procyonidae", "procyon", ""),
    "opossum": ("mammalia", "didelphimorphia", "didelphidae", "", ""),
    "rabbit": ("mammalia", "lagomorpha", "leporidae", "", ""),
    "skunk": ("mammalia", "carnivora", "mephitidae", "", ""),
    "deer": ("mammalia", "artiodactyla", "cervidae", "", ""),
    "squirrel": ("mammalia", "rodentia", "sciuridae", "", ""),
    "rodent": ("mammalia", "rodentia", "", "", ""),
    "bird": ("aves", "", "", "", ""),
}
NO_RESULT = "no cv result"


def parse_taxon(label: str) -> dict:
    """'uuid;class;order;family;genus;species;common name' -> dict of the parts."""
    parts = label.split(";")
    parts += [""] * (7 - len(parts))
    return {"class": parts[1], "order": parts[2], "family": parts[3], "genus": parts[4],
            "species": parts[5], "common": parts[6]}


def map_taxon(label: str) -> str:
    """SpeciesNet label -> one of the classes, or 'coarse: <common name>' / the common name."""
    if NO_RESULT in label:  # the classifier produced nothing for this box
        return NO_DETECTION
    t = parse_taxon(label)
    for levels, values, cls in TAXON_RULES:
        if tuple(t[level] for level in levels) == values:
            return cls
    return f"other: {t['common']}"


def image_label(prediction: dict) -> tuple[str, str]:
    """(SpeciesNet label, mapped class) of the most confident classified detection."""
    for det in sorted(prediction.get("detections", []), key=lambda d: -d["conf"]):
        if "processed_classification" in det:
            raw = det["processed_classification"]["class"]
            return raw, map_taxon(raw)
    return "", NO_DETECTION


def is_coarser_match(raw: str, true_class: str) -> bool:
    """True if raw is a higher taxon that contains true_class (e.g. canidae for coyote).

    Only labels at taxonomic class level or below count ("animal" does not).
    """
    t = parse_taxon(raw)
    ref = dict(zip(LEVELS, REFERENCE.get(true_class, ("",) * 5)))
    filled = [lvl for lvl in LEVELS if t[lvl]]
    deepest_ref = max((i for i, lvl in enumerate(LEVELS) if ref[lvl]), default=-1)
    return (bool(filled) and len(filled) - 1 < deepest_ref
            and all(t[lvl] == ref[lvl] for lvl in filled))


def burst_vote(labels: pd.Series, seq_ids: pd.Series, classes: list[str]) -> pd.Series:
    """Majority class per burst over frames with a mapped class; frames keep their own label otherwise."""
    df = pd.DataFrame({"label": labels.values, "seq": seq_ids.values}, index=labels.index)
    votes = df[df.label.isin(classes)].groupby("seq").label.agg(lambda s: Counter(s).most_common(1)[0][0])
    return df.seq.map(votes).fillna(df.label)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=Path("data/speciesnet/predictions.json"))
    parser.add_argument("--own", type=Path, default=Path("data/predictions/classify_test.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/06_speciesnet.md"))
    args = parser.parse_args(argv)

    classes = load_classes()
    preds = json.loads(args.predictions.read_text())["predictions"]
    sn = pd.DataFrame([{"file_name": Path(p["filepath"]).name, **dict(zip(("raw", "speciesnet"), image_label(p)))}
                       for p in preds])
    own = pd.read_csv(args.own)
    own = own[own.evaluation.str.startswith("cv_fold")]
    images, _ = load_cct()
    own = own.merge(images[["image_id", "file_name"]], on="image_id")
    df = own.merge(sn, on="file_name")
    if len(df) != len(sn):
        raise RuntimeError("SpeciesNet predictions do not match the test images")
    df["speciesnet_burst"] = burst_vote(df.speciesnet, df.seq_id, classes)

    wrong = df[df.speciesnet != df.label]
    coarse = wrong.apply(lambda r: is_coarser_match(r.raw, r.label), axis=1)

    cols = {"speciesnet": "SpeciesNet v4.0.3a (geofence USA/CA)",
            "speciesnet_burst": "SpeciesNet + burst majority vote",
            "full_image": "own: full image (baseline)",
            "crops_image": "own: MegaDetector crops",
            "crops_sequence": "own: crops + burst average"}
    summary = pd.DataFrame({name: metrics.summary(df.label, df[col], classes) for col, name in cols.items()}).T
    per_class = pd.concat({
        "SpeciesNet F1": metrics.per_class(df.label, df.speciesnet, classes).f1,
        "own crops F1": metrics.per_class(df.label, df.crops_image, classes).f1,
        "own crops + burst F1": metrics.per_class(df.label, df.crops_sequence, classes).f1,
        "support": metrics.per_class(df.label, df.speciesnet, classes).support,
    }, axis=1)
    per_cam = df.assign(speciesnet=df.label == df.speciesnet, own_crops=df.label == df.crops_image,
                        own_crops_burst=df.label == df.crops_sequence) \
        .groupby("location")[["speciesnet", "own_crops", "own_crops_burst"]].mean()
    top_wrong = wrong.assign(pred=wrong.raw.map(lambda r: parse_taxon(r)["common"] or "(no detection)")) \
        .groupby(["label", "pred"]).size().sort_values(ascending=False).head(15)

    lines = [
        "# SpeciesNet vs. own classifier", "",
        "Generated by `python -m mammalcam.speciesnet.evaluate`. Do not edit by hand.", "",
        f"- Test images: {len(df):,} (same as for the own models; own predictions from the camera "
        "cross-validation, so no model saw the camera of a test image)",
        "- SpeciesNet: v4.0.3a crop classifier on the same MegaDetector boxes, its own ensemble with "
        "geofence (country USA, state CA) and taxonomic roll-up; image prediction = most confident detection",
        "- SpeciesNet labels are mapped to the 12 classes by taxonomy (e.g. family Leporidae → rabbit). "
        "Anything else counts as wrong.",
        f"- Of SpeciesNet's wrong predictions, {coarse.sum():,} ({coarse.sum() / len(df):.1%} of all images) are "
        "correct but coarser than the class, e.g. 'canine family' for a coyote.",
        "- Caveat: SpeciesNet's training data is not published per dataset; Caltech Camera Traps is public "
        "and may have been part of it.", "",
        "## Overall", "",
        summary[["n", "accuracy", "macro_f1"]].to_markdown(floatfmt=".3f"), "",
        "## Per class (F1)", "",
        per_class.sort_values("support", ascending=False).to_markdown(floatfmt=".3f"), "",
        "## Accuracy per camera", "",
        per_cam.to_markdown(floatfmt=".3f"), "",
        "## Most frequent SpeciesNet errors (true class → SpeciesNet label)", "",
        top_wrong.rename("images").to_frame().to_markdown(), "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    Path("reports/results").mkdir(parents=True, exist_ok=True)
    Path("reports/results/speciesnet.json").write_text(json.dumps(
        {"summary": summary.to_dict(orient="index"), "coarser_but_correct": int(coarse.sum())}, indent=2))
    df[["image_id", "location", "label", "raw", "speciesnet", "speciesnet_burst"]] \
        .to_csv("data/predictions/speciesnet_test.csv", index=False)
    print(summary[["accuracy", "macro_f1"]].round(3))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
