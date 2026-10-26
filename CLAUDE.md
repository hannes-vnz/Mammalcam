# CLAUDE.md

Portfolio project: camera trap images → empty filter → MegaDetector boxes → own species classifier → CSV + Gradio demo, linked to Mammalatlas.

## Rules

- **Location split is the core.** Train/val/test are separated by camera location, never randomly. A random split exists only as the comparison that shows overestimation. Tests must guarantee that no location appears in two splits.
- **MegaDetector is pretrained and used for inference only.** The own work is the stage-2 species classifier (transfer learning via timm).
- **No data in the repo.** Only download scripts. Before any download, state dataset, source, licence, size in GB and number of images, and wait for the user's OK.
- **No keys or credentials in the repo** (.env, kaggle.json are gitignored).
- **Open-source tools only.**
- **Work in small, checkable steps:** dataset + location split → MegaDetector on a sample → crops → baseline → classifier → evaluation → demo. Stop for review after each.
- **Tests** for data preparation: split without location overlap, crops, label mapping.
- **Commits** in meaningful units with clear messages. Never push, the user pushes.
- **Honest results:** report limitations, rare species and failure cases in the README.
- Demo example images only from the free dataset.

## Environment

- MacBook Air (Apple Silicon). Project-local `.venv`, nothing global. Python ≥ 3.10 (system Python 3.9 is too old for current Gradio/PyTorch).
- Local inference on MPS or CPU. Real fine-tuning runs in a notebook on Kaggle or Colab (free GPU).

## Mammalatlas link

`https://hannes-vnz.github.io/Mammalatlas/#art=<scientific name, URL-encoded>`; the atlas matches the name case-insensitively. Only for mammals with an unambiguous scientific name; coarse classes (rodent, bird) get no link.

## Language

Code, README and commits in English. Talk to the user in German.
