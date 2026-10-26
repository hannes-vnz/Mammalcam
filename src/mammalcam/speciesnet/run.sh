#!/bin/sh
# Run Google's SpeciesNet (v4.0.3a, crop classifier) on the CCT20 test images,
# reusing LILA's MegaDetector boxes instead of running the detector again.
#
# Needs: python -m mammalcam.speciesnet.prepare (main .venv) and a separate
#   uv venv --python 3.12 .venv-speciesnet && uv pip install --python .venv-speciesnet/bin/python speciesnet
# Run from the repository root:  sh src/mammalcam/speciesnet/run.sh
# Both steps resume where they stopped if interrupted.
set -e
cd data/speciesnet
PY=../../.venv-speciesnet/bin/python

# 1. classifier on every MegaDetector box (about 3 hours on an M-series MacBook)
caffeinate -i $PY -m speciesnet.scripts.run_model --classifier_only \
  --instances_json instances.json --detections_json detections.json \
  --predictions_json classifications.json --batch_size 32 --noprogress_bars --bypass_prompts

# 2. SpeciesNet's own ensemble: geofence (USA/CA from instances.json) and taxonomic roll-up
$PY -m speciesnet.scripts.run_model --ensemble_only \
  --instances_json instances.json --detections_json detections.json \
  --classifications_json classifications.json --predictions_json predictions.json \
  --noprogress_bars --bypass_prompts
