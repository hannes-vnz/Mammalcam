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

To be filled in once the dataset is fixed. Datasets come from [LILA BC](https://lila.science) and are not part of this repository; download scripts are in `src/mammalcam/data/`.

## Results

Pending.

## Limitations

Pending.

## Run it

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[detect,demo,dev]"
pytest
```

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
