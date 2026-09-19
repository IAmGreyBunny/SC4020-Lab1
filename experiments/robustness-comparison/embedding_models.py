"""Model construction and embedding evaluation."""

import numpy as np


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name, device):
    import torch
    from torchvision import models as tv_models

    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT
        model = tv_models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
    elif name == "vit":
        weights = tv_models.ViT_B_16_Weights.DEFAULT
        model = tv_models.vit_b_16(weights=weights)
        model.heads = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown model: {name}")
    model.eval().to(device)
    return model, weights.transforms()


def embed_images(images, model, preprocess, device, batch_size):
    import torch
    import torch.nn.functional as F

    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(image) for image in batch]).to(device)
        with torch.inference_mode():
            vectors = F.normalize(model(tensors).float(), p=2, dim=1)
        features.append(vectors.cpu().numpy())
    return np.concatenate(features, axis=0)


def run_model(model_name, variants, device, batch_size):
    model, preprocess = build_model(model_name, device)
    embeddings = embed_images(
        [variant["image"] for variant in variants],
        model,
        preprocess,
        device,
        batch_size,
    )
    if embeddings.ndim != 2:
        raise RuntimeError(f"{model_name} produced unexpected shape {embeddings.shape}")

    original_rows = {
        variant["query"]: row
        for row, variant in enumerate(variants)
        if variant["transform"] == "original"
    }
    rows = []
    for row, variant in enumerate(variants):
        if variant["transform"] == "original":
            continue
        original = embeddings[original_rows[variant["query"]]]
        candidate = embeddings[row]
        cosine = float(np.dot(original, candidate) / (
            np.linalg.norm(original) * np.linalg.norm(candidate)
        ))
        rows.append({
            "model": model_name,
            "query": variant["query"],
            "image_id": variant["image_id"],
            "transform": variant["transform"],
            "level": variant["value"],
            "label": variant["label"],
            "cosine": round(cosine, 6),
        })
    return rows
