"""Registry of the 2x2 model x dataset configurations.

Shared by benchmark_ann.py and compare_methods.ipynb. Paths are absolute,
resolved from the repository root. Every configuration uses the same
benchmark protocol; only the data-forced settings (dimension, PQ
sub-quantizers) differ. Embeddings come from pipeline/extract_embeddings.py.

Both models on one dataset have the same row count and order, so the
seed-42 query/gallery split - and therefore the query set - is identical
across models. That is what makes the model comparison fair.
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


MODELS = {
    # FAISS IVF-PQ needs dimension % pq_m == 0: 768 / 48 and 512 / 32.
    "vit": {"label": "ViT-B/16", "dimension": 768, "pq_m": 48},
    "resnet": {"label": "ResNet18", "dimension": 512, "pq_m": 32},
}

DATASETS = {
    "coco": {
        "label": "COCO",
        "data_dir": REPO_ROOT / "data" / "circo",
        "image_dir": _coco_image_dir(),
    },
    "zappos": {
        "label": "Zappos",
        "data_dir": REPO_ROOT / "data" / "zappos",
        "image_dir": REPO_ROOT / "data" / "zappos" / "ut-zap50k-images" / "ut-zap50k-images",
    },
}

CONFIGS = {}
for _model, _m in MODELS.items():
    for _dataset, _d in DATASETS.items():
        CONFIGS[f"{_model}-{_dataset}"] = {
            "model": _model,
            "dataset": _dataset,
            "label": f"{_m['label']} / {_d['label']}",
            "emb": _d["data_dir"] / f"{_model}_embeddings.npy",
            "ids": _d["data_dir"] / f"{_model}_image_ids.npy",
            "image_dir": _d["image_dir"],
            "dimension": _m["dimension"],
            "pq_m": _m["pq_m"],
        }


def results_dir(key):
    return RESULTS_DIR / key
