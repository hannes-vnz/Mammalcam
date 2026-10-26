"""Small, class-balanced image samples for quick checks (e.g. MegaDetector)."""

from __future__ import annotations

import pandas as pd


def stratified_sample(images: pd.DataFrame, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """About ``n`` images, as evenly spread over labels as the data allows.

    At most one image per sequence, so near-identical burst frames do not
    inflate the sample. Rare labels contribute everything they have; the
    remaining budget goes to the other labels in equal parts.
    """
    one_per_seq = images.sample(frac=1, random_state=seed).drop_duplicates("seq_id")
    groups = {label: g for label, g in one_per_seq.groupby("label")}
    quota = dict.fromkeys(groups, 0)
    budget = min(n, len(one_per_seq))
    # hand out the budget one round at a time until it is spent
    while budget > 0:
        open_labels = [k for k in groups if quota[k] < len(groups[k])]
        if not open_labels:
            break
        share = max(1, budget // len(open_labels))
        for k in open_labels:
            take = min(share, len(groups[k]) - quota[k], budget)
            quota[k] += take
            budget -= take
            if budget == 0:
                break
    parts = [groups[k].head(q) for k, q in quota.items() if q]
    return pd.concat(parts).sort_values(["label", "image_id"]).reset_index(drop=True)
