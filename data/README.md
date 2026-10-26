# data/

Not tracked by git. Datasets are fetched with the download scripts in `src/mammalcam/data/`.

Expected layout (created by the scripts):

```
data/
├── raw/        # images and metadata as downloaded from LILA BC
├── splits/     # train/val/test lists (location-based and random)
├── detections/ # MegaDetector output (JSON)
└── crops/      # animal crops for the classifier
```
