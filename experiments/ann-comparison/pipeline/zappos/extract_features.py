"""Generate ResNet18 embeddings for the UT-Zap50K image collection.

The output contract mirrors the team's ViT embedding pipeline:

- ``<prefix>_embeddings.npy``: float32, shape (N, 512), L2-normalized
- ``<prefix>_image_ids.npy``: int64, shape (N,), row-aligned with embeddings

Images are processed in lexicographically sorted path order. UT-Zap50K does
not expose a convenient COCO-style integer ID, so IDs are deterministic row
numbers (0 through N-1) in that sorted order.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from common import IMAGE_ROOT, prepare_data


DIMENSION = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_OUTPUT_DIR = Path("data/zappos")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ShoeDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index]["image"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor


def create_model(device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def validate_outputs(embeddings_path, image_ids_path):
    embeddings = np.load(embeddings_path, mmap_mode="r")
    image_ids = np.load(image_ids_path, mmap_mode="r")
    norms = np.linalg.norm(embeddings, axis=1)

    print(f"Embeddings: {embeddings.shape}")
    print(f"Image IDs: {image_ids.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Image ID dtype: {image_ids.dtype}")
    print(f"Row counts match: {len(embeddings) == len(image_ids)}")
    print(f"L2 normalized: {np.allclose(norms, 1.0, atol=1e-3)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate L2-normalized ResNet18 embeddings for UT-Zap50K."
    )
    parser.add_argument("--images-dir", default=str(IMAGE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="resnet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    image_root = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir)

    dataframe = prepare_data(limit=args.limit, image_root=image_root)
    if dataframe.empty:
        raise RuntimeError(
            f"No Zappos JPG images found. Expected them under {image_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{args.prefix}_embeddings.npy"
    image_ids_path = output_dir / f"{args.prefix}_image_ids.npy"

    device = get_device()
    model, transform = create_model(device)
    dataset = ShoeDataset(dataframe, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Images: {len(dataset)}")
    print(f"Device: {device}")
    print(f"Image root: {image_root}")

    batches = []
    with torch.inference_mode():
        for batch_number, images in enumerate(loader, start=1):
            features = model(images.to(device))
            features = F.normalize(features.float(), p=2, dim=1)
            batches.append(features.cpu())
            print(f"Batch {batch_number}/{len(loader)} completed", flush=True)

    embeddings = (
        torch.cat(batches)
        .numpy()
        .astype(np.float32, copy=False)
    )
    if embeddings.shape != (len(dataset), DIMENSION):
        raise RuntimeError(
            f"Unexpected embedding shape {embeddings.shape}; "
            f"expected ({len(dataset)}, {DIMENSION})"
        )

    image_ids = np.arange(len(dataset), dtype=np.int64)
    np.save(embeddings_path, embeddings)
    np.save(image_ids_path, image_ids)

    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Image IDs saved to: {image_ids_path}")
    validate_outputs(embeddings_path, image_ids_path)


if __name__ == "__main__":
    main()
