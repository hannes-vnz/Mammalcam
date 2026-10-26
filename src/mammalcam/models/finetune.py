"""Fine-tune the backbone on MegaDetector crops, with the same camera cross-validation.

Usage (on a GPU, e.g. Colab or Kaggle; see notebooks/finetune_colab_kaggle.ipynb):
    python -m mammalcam.models.finetune --epochs 6 --out outputs/finetune

Same classes, crops, camera groups and end-to-end test as the linear probe in
mammalcam.models.classify, so the results are directly comparable:

- train on the labelled crops (crops.csv) of the training cameras
- pick the epoch with the best crop-level macro-F1 on the validation cameras
- predict every MegaDetector detection of the test cameras (detections.csv),
  average per image (weighted by detection confidence) and per burst

Each fold writes its predictions to <out>/fold_<k>.csv as soon as it is
done, so an interrupted run resumes with the next fold. Afterwards
<out>/predictions.csv holds all folds in the format of
data/predictions/classify_test.csv (model "finetuned").
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

from mammalcam.data.cct import DEFAULT_ANN_DIR, load_cct
from mammalcam.data.split import camera_folds, fold_parts
from mammalcam.models.baseline import load_classes, select_images
from mammalcam.models.classify import image_proba_from_crops, sequence_mean, to_labels
from mammalcam.models.features import DEFAULT_BACKBONE, device

CROPS_DIR = Path("data/crops")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def transforms(train: bool, size: int = 224):
    from torchvision import transforms as T

    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.2),
            T.RandomGrayscale(p=0.2),  # many test cameras shoot infrared at night
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    # same squashing resize as the linear probe's features
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class Crops(torch.utils.data.Dataset):
    def __init__(self, files, labels, root: Path, transform):
        self.files, self.labels = list(files), list(labels)
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.root / self.files[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i]


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """'Balanced' weights as in scikit-learn; classes missing from y get weight 0."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def predict(model, loader, dev) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out.append(torch.softmax(model(x.to(dev, non_blocking=True)).float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.num_classes))


def train_fold(train: pd.DataFrame, val: pd.DataFrame, classes: list[str], cfg: dict, dev) -> tuple:
    """Fine-tune on train crops, select the epoch by val macro-F1. Returns (model, history)."""
    import timm

    idx = {c: i for i, c in enumerate(classes)}
    y_train = train.label.map(idx).values
    y_val = val.label.map(idx).values
    model = timm.create_model(cfg["backbone"], pretrained=cfg["pretrained"], num_classes=len(classes),
                              drop_path_rate=0.1).to(dev)
    loaders = {
        "train": torch.utils.data.DataLoader(
            Crops(train.crop_file, y_train, cfg["crops_dir"] / "images", transforms(True, cfg["size"])),
            batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"], drop_last=True,
            pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0),
        "val": torch.utils.data.DataLoader(
            Crops(val.crop_file, y_val, cfg["crops_dir"] / "images", transforms(False, cfg["size"])),
            batch_size=cfg["batch_size"] * 2, num_workers=cfg["workers"]),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.05)
    total = cfg["epochs"] * len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, total, warmup=max(1, len(loaders["train"]) // 2)))
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(y_train, len(classes)).to(dev), label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    best, best_f1, history = None, -1.0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for step, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if cfg["max_steps"] and step + 1 >= cfg["max_steps"]:
                break
        proba = predict(model, loaders["val"], dev)
        f1 = (f1_score(y_val, proba.argmax(1), average="macro", labels=np.unique(y_train), zero_division=0)
              if len(y_val) else 0.0)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_f1": float(f1),
                        "seconds": round(time.time() - t0)})
        print(f"  epoch {epoch + 1}: loss {np.mean(losses):.3f}, val macro-F1 {f1:.3f}, {time.time() - t0:.0f}s",
              flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, history


def run_fold(fold: int, k: int, folds: dict, images: pd.DataFrame, tables: dict, classes: list[str], cfg: dict,
             dev) -> tuple[pd.DataFrame, list]:
    """Train on one camera split and predict its test images end to end."""
    part = images.set_index("image_id").location.map(fold_parts(folds, fold, k))
    crops, dets, test_images = tables["crops"], tables["dets"], tables["test_images"]
    cpart = crops.image_id.map(part).values
    train, val = crops[cpart == "train"], crops[cpart == "val"]
    if cfg["limit"]:
        train = train.sample(min(cfg["limit"], len(train)), random_state=0)
        val = val.sample(min(cfg["limit"] // 4 + 1, len(val)), random_state=0)
    print(f"fold {fold}: {len(train)} train crops, {len(val)} val crops", flush=True)
    model, history = train_fold(train, val, classes, cfg, dev)

    test = test_images[test_images.image_id.map(part).values == "test"]
    if cfg["limit"]:
        test = test.sample(min(cfg["limit"], len(test)), random_state=0)
    d = dets[dets.image_id.isin(test.image_id)]
    loader = torch.utils.data.DataLoader(
        Crops(d.crop_file, np.zeros(len(d), dtype=int), cfg["crops_dir"] / "images", transforms(False, cfg["size"])),
        batch_size=cfg["batch_size"] * 2, num_workers=cfg["workers"])
    crop_proba = predict(model, loader, dev)
    proba = image_proba_from_crops(crop_proba, d.image_id, d.conf, test.image_id)
    out = test[["image_id", "seq_id", "location", "label"]].copy()
    out["crops_image"] = to_labels(proba, classes)
    out["crops_sequence"] = to_labels(sequence_mean(proba, test.seq_id), classes)
    out["evaluation"] = f"cv_fold_{fold}"
    out["model"] = "finetuned"
    return out, history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--crops-dir", type=Path, default=CROPS_DIR)
    parser.add_argument("--out", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--k", type=int, default=5, help="number of camera groups")
    parser.add_argument("--folds", default=None, help="comma-separated subset of folds to run, e.g. 0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: use only this many crops per fold")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: stop each epoch after N steps")
    parser.add_argument("--no-pretrained", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    dev = device()
    cfg = {"backbone": args.backbone, "pretrained": not args.no_pretrained, "epochs": args.epochs,
           "batch_size": args.batch_size, "lr": args.lr, "size": args.size, "workers": args.workers,
           "limit": args.limit, "max_steps": args.max_steps, "crops_dir": args.crops_dir}
    print(f"device {dev}, backbone {args.backbone}", flush=True)

    classes = load_classes()
    images, _ = load_cct(args.ann_dir)
    folds = camera_folds(images, args.k, args.seed)
    reference = Path("reports/results/classify.json")
    if reference.exists():  # the camera groups must be the ones of the linear probe
        groups = json.loads(reference.read_text())["cv"]["camera_groups"]
        if {int(loc): int(f) for f, locs in groups.items() for loc in locs} != folds:
            raise RuntimeError("camera groups differ from reports/results/classify.json")
    crops = pd.read_csv(args.crops_dir / "crops.csv")
    tables = {
        "crops": crops[crops.label.isin(classes)],
        "dets": pd.read_csv(args.crops_dir / "detections.csv"),
        "test_images": select_images(images, classes),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    todo = range(args.k) if args.folds is None else [int(f) for f in args.folds.split(",")]
    for fold in todo:
        path = args.out / f"fold_{fold}.csv"
        if path.exists():
            print(f"fold {fold}: already done, skipping")
            continue
        preds, history = run_fold(fold, args.k, folds, images, tables, classes, cfg, dev)
        preds.to_csv(path, index=False)
        (args.out / f"fold_{fold}_history.json").write_text(json.dumps(history, indent=2))
        acc = (preds.crops_sequence == preds.label).mean()
        print(f"fold {fold}: test accuracy (crops + burst) {acc:.3f} on {len(preds)} images", flush=True)

    done = sorted(args.out.glob("fold_*.csv"))
    pd.concat(pd.read_csv(p) for p in done).to_csv(args.out / "predictions.csv", index=False)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))
    print(f"{len(done)}/{args.k} folds done -> {args.out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
