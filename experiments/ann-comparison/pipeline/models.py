"""Shared model construction and image ordering for every experiment.

`pipeline/extract_embeddings.py` (full-dataset extraction) and
`pipeline/corruption.py` (corrupted-query robustness) must build the exact
same feature extractors and enumerate images in the exact same order, or
their outputs are not comparable. Both import this module, so the weights,
preprocessing and row ordering have a single definition.

- models   : torchvision pretrained weights (DEFAULT), classifier removed
             (ViT heads / ResNet fc -> Identity), weights.transforms()
- ordering : COCO sorted by filename, id = numeric stem;
             Zappos sorted by full path, id = row number 0..N-1

Embedding row i always corresponds to `list_images(...)[0][i]`.
"""

from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

MODELS = {
    "vit": {"dimension": 768, "batch_size": 64},
    "resnet": {"dimension": 512, "batch_size": 128},
}

DATASETS = {
    "coco": {
        "images_dir": REPO_ROOT / "data" / "circo",
        "output_dir": REPO_ROOT / "data" / "circo",
    },
    "zappos": {
        "images_dir": REPO_ROOT / "data" / "zappos" / "ut-zap50k-images" / "ut-zap50k-images",
        "output_dir": REPO_ROOT / "data" / "zappos",
    },
}


def create_model(name):
    """Return (model, preprocess, weights_name) for 'vit' or 'resnet'."""
    from torchvision import models

    if name == "vit":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        # Drop the classifier head: the model returns the 768-d embedding.
        model.heads = torch.nn.Identity()
        weights_name = "ViT_B_16_Weights.DEFAULT"
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        # Drop the classifier: the model returns 512-d avgpool features.
        model.fc = torch.nn.Identity()
        weights_name = "ResNet18_Weights.DEFAULT"
    return model.eval(), weights.transforms(), weights_name


def list_images(dataset, images_dir):
    """Return (paths, ids) in the canonical order used for the embedding rows."""
    images_dir = Path(images_dir)
    if dataset == "coco":
        # COCO filenames are zero-padded ids, so filename order == id order.
        paths = sorted(
            (p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")),
            key=lambda p: p.name,
        )
        ids = [int(p.stem) for p in paths]
    else:
        # UT-Zap50K has no integer ids: sort by full path string and number
        # the rows, exactly as the team's original ResNet/Zappos extractor did.
        paths = sorted(images_dir.rglob("*.jpg"), key=lambda p: str(p))
        ids = list(range(len(paths)))
    return paths, ids


def embed_images(images, model, preprocess, device, batch_size):
    """Embed PIL images to L2-normalized float32 vectors, batched on device."""
    import torch.nn.functional as F

    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(image) for image in batch]).to(device)
        with torch.inference_mode():
            vectors = F.normalize(model(tensors).float(), p=2, dim=1)
        features.append(vectors.cpu().numpy())
    return np.concatenate(features, axis=0)
