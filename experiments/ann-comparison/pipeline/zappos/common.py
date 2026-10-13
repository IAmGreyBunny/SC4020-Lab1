"""UT-Zap50K dataset discovery shared by the ResNet18 scripts."""

from pathlib import Path

import pandas as pd


# Local default. Override it with extract_features.py --images-dir when needed.
ZAPPOS_DATASET_ROOT = Path(
    "/Users/kohjiaxin/Y4S1/ResNet18/image-similarity-search/"
    "data/zappos/ut-zap50k-images/ut-zap50k-images"
)

# Backwards-compatible name used by search.py and evaluate.py.
IMAGE_ROOT = ZAPPOS_DATASET_ROOT


def prepare_data(limit=None, image_root=IMAGE_ROOT):
    """Return image paths and path-derived labels in stable sorted order."""
    image_root = Path(image_root)
    image_paths = sorted(
        image_root.rglob("*.jpg"),
        key=lambda path: str(path),
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    records = []
    for path in image_paths:
        parts = path.relative_to(image_root).parts
        records.append(
            {
                "image": str(path),
                "filename": path.name,
                "category": parts[0],
                "subcategory": parts[1],
                "brand": parts[2],
            }
        )

    return pd.DataFrame(records)
