"""Image loading helpers for folders and ZIP archives."""

import io
import random
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _image_count(path):
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            return sum(
                item.filename.lower().endswith(IMAGE_SUFFIXES)
                for item in archive.infolist()
            )
    return sum(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        for item in path.rglob("*")
    )


def discover_datasets(data_root=DEFAULT_DATA_ROOT):
    """Return image directories and ZIP archives keyed by relative dataset name."""
    root = Path(data_root)
    if not root.is_dir():
        return {}

    directory_candidates = set()
    archive_candidates = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".zip":
            if _image_count(path):
                archive_candidates[path.relative_to(root).as_posix()] = path
        elif path.is_dir() and _image_count(path):
            directory_candidates.add(path)

    leaf_directories = {
        path for path in directory_candidates
        if not any(path != child and path in child.parents for child in directory_candidates)
    }
    datasets = {
        path.relative_to(root).as_posix(): path for path in leaf_directories
    }
    datasets.update(archive_candidates)
    return dict(sorted(datasets.items()))


def resolve_dataset(name, data_root=DEFAULT_DATA_ROOT):
    """Resolve a discovered dataset name or raise a helpful error."""
    datasets = discover_datasets(data_root)
    if name not in datasets:
        available = ", ".join(datasets) or "none"
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")
    return datasets[name]


def format_datasets(datasets):
    """Format discovered datasets for the command-line listing."""
    if not datasets:
        return "No image directories or ZIP archives found."
    return "\n".join(f"  {name}" for name in datasets)


def load_from_zip(zip_path, count, seed):
    rng = random.Random(seed)
    path = Path(zip_path)
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            item.filename
            for item in archive.infolist()
            if item.filename.lower().endswith((".jpg", ".jpeg"))
        )
        if not names:
            raise RuntimeError(f"No JPG images found in {path}")
        chosen = rng.sample(names, min(count, len(names)))
        images = []
        for name in chosen:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                images.append(image.convert("RGB"))
    return images, chosen


def load_from_dir(image_dir, count, seed):
    rng = random.Random(seed)
    root = Path(image_dir)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No JPG/PNG images found under {root}")
    chosen = rng.sample(paths, min(count, len(paths)))
    images = []
    for path in chosen:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images, [str(path) for path in chosen]
