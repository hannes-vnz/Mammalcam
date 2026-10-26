import pandas as pd
import pytest

from mammalcam.models.classify import NO_DETECTION
from mammalcam.speciesnet.evaluate import burst_vote, image_label, is_coarser_match, map_taxon
from mammalcam.speciesnet.prepare import to_speciesnet_detections

U = "00000000-0000-0000-0000-000000000000"
CLASSES = ["bobcat", "coyote", "rabbit", "rodent", "squirrel"]


@pytest.mark.parametrize("label, expected", [
    (f"{U};mammalia;carnivora;felidae;lynx;rufus;bobcat", "bobcat"),
    (f"{U};mammalia;carnivora;canidae;canis;latrans;coyote", "coyote"),
    (f"{U};mammalia;carnivora;canidae;canis;familiaris;domestic dog", "dog"),
    (f"{U};mammalia;lagomorpha;leporidae;;;rabbit and hare family", "rabbit"),
    (f"{U};mammalia;rodentia;sciuridae;sciurus;carolinensis;eastern gray squirrel", "squirrel"),
    (f"{U};mammalia;rodentia;cricetidae;neotoma;;neotoma species", "rodent"),
    (f"{U};aves;galliformes;odontophoridae;callipepla;californica;california quail", "bird"),
    (f"{U};mammalia;artiodactyla;cervidae;odocoileus;hemionus;mule deer", "deer"),
    (f"{U};mammalia;carnivora;canidae;urocyon;cinereoargenteus;grey fox", "other: grey fox"),
    (f"{U};mammalia;carnivora;canidae;;;canine family", "other: canine family"),
    (f"{U};;;;;;blank", "other: blank"),
    ("no cv result;no cv result;no cv result;no cv result;no cv result;no cv result", NO_DETECTION),
])
def test_map_taxon(label, expected):
    assert map_taxon(label) == expected


@pytest.mark.parametrize("raw, true, expected", [
    (f"{U};mammalia;carnivora;canidae;;;canine family", "coyote", True),
    (f"{U};mammalia;;;;;mammal", "rodent", True),
    (f"{U};mammalia;carnivora;felidae;;;cat family", "coyote", False),       # wrong family
    (f"{U};mammalia;rodentia;sciuridae;;;sciuridae family", "rodent", False),  # squirrel, not coarser
    (f"{U};;;;;;animal", "coyote", False),                                   # too coarse to count
    (f"{U};mammalia;lagomorpha;leporidae;;;rabbit and hare family", "rabbit", False),  # exact, not coarser
])
def test_is_coarser_match(raw, true, expected):
    assert is_coarser_match(raw, true) is expected


def test_image_label_uses_most_confident_classified_detection():
    pred = {"detections": [
        {"conf": 0.3, "processed_classification": {"class": f"{U};mammalia;carnivora;felidae;lynx;rufus;bobcat"}},
        {"conf": 0.9, "processed_classification": {"class": f"{U};mammalia;carnivora;canidae;canis;latrans;coyote"}},
        {"conf": 0.95, "label": "human"},  # not classified
    ]}
    assert image_label(pred)[1] == "coyote"
    assert image_label({"detections": []}) == ("", NO_DETECTION)


def test_burst_vote_takes_majority_and_fills_unmapped_frames():
    labels = pd.Series(["coyote", "coyote", "bobcat", "other: canine family", "rabbit"])
    seqs = pd.Series(["s1", "s1", "s1", "s1", "s2"])
    assert burst_vote(labels, seqs, CLASSES).tolist() == ["coyote"] * 4 + ["rabbit"]


def test_to_speciesnet_detections_converts_and_filters():
    md = {"images": [
        {"file": "a.jpg", "detections": [{"category": "2", "conf": 0.5, "bbox": [0, 0, 1, 1]},
                                         {"category": "1", "conf": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2]}]},
        {"file": "not_in_test.jpg", "detections": []},
    ]}
    out = to_speciesnet_detections(md, {"a.jpg": "/data/a.jpg"})
    assert len(out["predictions"]) == 1
    p = out["predictions"][0]
    assert p["filepath"] == "/data/a.jpg"
    assert [d["label"] for d in p["detections"]] == ["animal", "human"]  # sorted by confidence
