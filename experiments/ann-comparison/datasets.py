"""Dataset registry shared by benchmark_ann.py and compare_methods.ipynb.

Paths are relative to the repository root. Each entry describes one
embedding set; the benchmark protocol is identical for all of them, only
the data-forced settings (dimension, PQ sub-quantizers) differ.
"""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"


def _coco_image_dir():
    # unzip_coco.py may extract into data/circo/unlabeled2017/ or straight
    # into data/circo/ depending on the zip layout; accept either.
    nested = REPO_ROOT / "data" / "circo" / "unlabeled2017"
    return nested if nested.exists() else REPO_ROOT / "data" / "circo"


DATASETS = {
    "coco": {
        "label": "ViT-COCO",
        "model": "ViT-B/16",
        "dimension": 768,
        "emb": REPO_ROOT / "data" / "circo" / "vit_embeddings.npy",
        "ids": REPO_ROOT / "data" / "circo" / "vit_image_ids.npy",
        "image_dir": _coco_image_dir(),
        # 768 % 48 == 0 -> 48 bytes per vector.
        "pq_m": 48,
    },
    "zappos": {
        "label": "ResNet18-Zappos",
        "model": "ResNet18",
        "dimension": 512,
        "emb": REPO_ROOT / "data" / "zappos" / "resnet_embeddings.npy",
        "ids": REPO_ROOT / "data" / "zappos" / "resnet_image_ids.npy",
        "image_dir": REPO_ROOT / "data" / "zappos" / "ut-zap50k-images" / "ut-zap50k-images",
        # FAISS requires d % m == 0 and 512 % 48 != 0, so 32 bytes per vector.
        "pq_m": 32,
    },
}


def results_dir(key):
    return RESULTS_DIR / key
