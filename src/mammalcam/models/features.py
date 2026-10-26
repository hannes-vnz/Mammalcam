"""Image embeddings from a frozen, pretrained timm backbone.

Used for the linear-probe baseline (full images) and the linear-probe crop
classifier, so both see exactly the same backbone and preprocessing apart
from the input. Embeddings are cached as .npy next to a CSV of the inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

DEFAULT_BACKBONE = "convnext_tiny.fb_in22k"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImageFiles(torch.utils.data.Dataset):
    """Loads images and resizes them to a square without cropping.

    A center crop would cut off animals at the image border, which are common
    in camera trap images; the slight aspect distortion is the lesser evil.
    """

    def __init__(self, paths: list[Path], size: int, mean, std):
        self.paths = list(paths)
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - self.mean) / self.std


@torch.no_grad()
def extract(paths: list[Path], backbone: str = DEFAULT_BACKBONE, batch_size: int = 64,
            workers: int = 4, pretrained: bool = True) -> np.ndarray:
    import timm

    dev = device()
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_model_data_config(model)
    ds = ImageFiles(paths, cfg["input_size"][-1], cfg["mean"], cfg["std"])
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers)
    out = []
    for i, x in enumerate(loader):
        out.append(model(x.to(dev)).float().cpu().numpy())
        print(f"\r{min((i + 1) * batch_size, len(ds))}/{len(ds)}", end="", flush=True)
    print()
    return np.concatenate(out) if out else np.zeros((0, model.num_features), dtype=np.float32)


def cached(name: str, table: pd.DataFrame, path_col: str, root: Path, cache_dir: Path = Path("data/features"),
           backbone: str = DEFAULT_BACKBONE, **kwargs) -> np.ndarray:
    """Embeddings for table[path_col] (relative to root), computed once per name."""
    npy, csv = cache_dir / f"{name}.npy", cache_dir / f"{name}.csv"
    keys = table[path_col].reset_index(drop=True)
    if npy.exists() and csv.exists() and pd.read_csv(csv)[path_col].equals(keys):
        return np.load(npy)
    feats = extract([root / p for p in keys], backbone, **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, feats)
    keys.to_frame().assign(backbone=backbone).to_csv(csv, index=False)
    return feats
