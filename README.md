# Mammalcam

Automatic analysis of camera trap images: filter out empty frames, find animals, identify the species, and link each result to its profile in the [Mammalatlas](https://github.com/hannes-vnz/Mammalatlas).

> **Status:** work in progress. Results below will be filled in as each step is completed.

## Problem and why it matters

Camera traps are one of the main tools for monitoring wildlife, but a single survey produces hundreds of thousands of images, most of them empty. Sorting them by hand takes ecologists weeks. Automating the first passes (empty or not, which species) frees that time for the actual conservation work.

The hard part is not reaching a high accuracy on a test set, but reaching it on **cameras the model has never seen**. Background, lighting and camera angle differ per location, and a model can learn the location instead of the animal. This project measures that effect explicitly.

## Pipeline

1. **Detection (stage 1):** [MegaDetector](https://github.com/agentmorris/MegaDetector), pretrained, finds animals, people and vehicles. Not retrained here.
2. **Crops:** each animal detection is cut out of the image.
3. **Species classification (stage 2):** transfer learning with a pretrained backbone (via [timm](https://github.com/huggingface/pytorch-image-models)) on the crops. This is the core of the project.
4. **Output:** CSV per image and a small web demo.

All animal classes in the dataset are classified. Mammals are the focus, and only mammals with an unambiguous scientific name get a Mammalatlas link. Low-confidence predictions are reported as "uncertain" instead of a forced species.

## Evaluation

- Train / validation / test split **by camera location**, never by random image.
- Comparison: the same model with a random split, to show how much accuracy would be overestimated.
- Accuracy, macro-F1, confusion matrix, per-species results (including rare species), error examples.
- Baseline: a classifier on full images without detector crops, to measure what stage 1 adds.

## Data and licences

**[Caltech Camera Traps, CCT20 benchmark subset](https://lila.science/datasets/caltech-camera-traps)**, hosted by LILA BC. Licence: [Community Data License Agreement – Permissive 1.0](https://cdla.dev/permissive-1-0/). Images are not part of this repository; `python -m mammalcam.data.download cct20` fetches them (6.5 GB).

> Beery, S., Van Horn, G., Perona, P. (2018). *Recognition in Terra Incognita.* ECCV 2018.

57,864 images from 20 camera locations in the southwestern US, 15 classes plus "empty". Full overview: [reports/01_dataset.md](reports/01_dataset.md).

What matters for this project:

- **Heavy class imbalance.** Opossum alone is 24 % of the images; badger (22 images) and fox (8) are too rare to learn.
- **Some classes exist at only two cameras** (car, deer, fox), so they cannot be in train, validation and test at the same time.
- **Only 7 % empty images.** The benchmark was pre-filtered; real deployments are often 70 % empty or more. The empty filter is therefore not tested under realistic conditions here.
- **Images come in bursts** of mostly three frames. Frames of one burst are near-duplicates.

### Splits

`python -m mammalcam.data.split` builds three splits with identical sizes (34,410 / 8,738 / 14,716 images):

| Strategy | What is separated | Shared cameras | Shared bursts |
|---|---|---|---|
| `location` (used for all results) | whole cameras: 12 train, 3 val, 5 test | 0 | 0 |
| `random_sequence` (comparison) | whole bursts | 20 | 0 |
| `random_image` (comparison) | single images | 20 | 13,395 |

The cameras for each part are chosen by a seeded search so that every class seen at two or more cameras appears in both train and test; the result is stored in [configs/splits_cct20.json](configs/splits_cct20.json).

## Results

### Stage 1: MegaDetector on a 500-image sample

MegaDetector v5a, pretrained, on a class-balanced sample (one image per burst, all 20 cameras). Full tables: [reports/02_megadetector_sample.md](reports/02_megadetector_sample.md).

- **Boxes:** where every animal has an annotated box, MegaDetector finds 97 % of them at threshold 0.2 (precision 97 %, mean IoU 0.85).
- **Image level:** it flags 87 % of the images labelled with an animal. But 50 of the 55 misses are frames without an annotated box. CCT labels whole bursts, so the last frame often shows the empty scene after the animal has left. On images with a box it misses 5 of 365.
- **"Empty" images:** 8 of 32 get a detection. Looking at them, most contain an animal after all (a close-up, a tail, a skunk at the edge) or a person. These are label errors, not detector errors.
- **Speed:** about 1 image per second on an M-series MacBook (MPS). The whole dataset would take about 16 hours locally.

Caveat: MegaDetector was trained on data that includes Caltech Camera Traps, so these numbers are probably better than on cameras it has never seen.

### Crops for the species classifier

To avoid 16 hours of local inference, the crops use [LILA's precomputed MegaDetector v5a results](https://lila.science/megadetector-results-for-camera-trap-datasets/) for Caltech Camera Traps (`python -m mammalcam.data.download cct-md`, 5 MB). They agree with the own run on the 500-image sample: 98 % same animal / no-animal decision, matched boxes at mean IoU 0.97.

`python -m mammalcam.crops.make_crops` turns every animal detection (confidence ≥ 0.2) into a padded square crop. Details: [reports/03_crops.md](reports/03_crops.md).

- **47,099 crops** from 45,039 images. 95 % of the labels are confirmed by an annotated box; detections on images with complete boxes that match none of them (rocks, branches) are dropped.
- **12 % of the animal images give no crop.** Most are burst frames after the animal has left; for rodents it is 33 %, because small animals are detected less often.
- **Hard cases for the classifier:** rodents have 101 training crops but 417 test crops, dogs only 41 validation crops. Fox (1 training crop) and badger (16) are too rare to train on.

## Limitations

Pending.

## Run it

Requires Python 3.10 or newer (the project pins 3.12 in `.python-version`). With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[detect,demo,dev]"
pytest
```

Plain `python3 -m venv .venv` and `pip install -e ".[detect,demo,dev]"` work as well.

## Project structure

```
configs/     # dataset, class list, class → scientific name for Mammalatlas links
src/mammalcam/
  data/      # download, metadata parsing, location-based split
  detect/    # MegaDetector inference
  crops/     # crops from detections, label assignment
  models/    # baseline and species classifier
  eval/      # metrics, confusion matrix, error examples
  export/    # CSV output
notebooks/   # exploration, MegaDetector sample, training notebook for Kaggle / Colab
app/         # Gradio demo
tests/       # tests for data preparation
data/        # not tracked, see data/README.md
```

## Licence

Code: MIT, see [LICENSE](LICENSE). Datasets keep their own licences (see above).
