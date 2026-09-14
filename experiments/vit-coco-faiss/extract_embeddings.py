"""Generate ViT-B/16 embeddings for the COCO unlabeled2017 images.

Adapted from the team's ViT_Embedding.ipynb (same model, weights and
preprocessing). Differences from the notebook, made for this experiment:

1. Batched GPU inference instead of one image at a time.
2. Deterministic order: images are processed sorted by filename.
3. Output is binary .npy (memmap) instead of CSV - smaller and much faster
   to load for FAISS.
4. L2-normalized vectors at save time so FAISS IndexFlatIP equals cosine
   similarity.
5. Resumable: progress is stored in the metadata JSON, and an interrupted
   run continues from the last completed row.
6. Can read images directly from the COCO zip (--zip) before/without
   extraction, or from an extracted folder (--images-dir).
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16

DIMENSION = 768
_local = threading.local()


def list_images_from_zip(zip_path):
    with ZipFile(zip_path) as archive:
        names = sorted(
            item.filename
            for item in archive.infolist()
            if item.filename.lower().endswith((".jpg", ".jpeg"))
        )
    return names


def list_images_from_dir(image_dir):
    return sorted(
        str(path)
        for path in Path(image_dir).rglob("*")
        if path.suffix.lower() in (".jpg", ".jpeg")
    )


def get_zip_handle(zip_path):
    # One handle per loader thread; a single ZipFile is not thread-safe.
    if not hasattr(_local, "handle"):
        _local.handle = ZipFile(zip_path)
    return _local.handle


def load_image(name, zip_path):
    if zip_path:
        data = get_zip_handle(zip_path).read(name)
        return Image.open(BytesIO(data))
    return Image.open(name)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip")
    source.add_argument("--images-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="vit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.zip:
        names = list_images_from_zip(args.zip)
        source_type, source_path = "zip", args.zip
    else:
        names = list_images_from_dir(args.images_dir)
        source_type, source_path = "dir", args.images_dir

    if args.limit is not None:
        names = names[: args.limit]
    total = len(names)

    # COCO filenames are zero-padded ids, e.g. 000000123456.jpg -> 123456.
    ids = [int(Path(name).stem) for name in names]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{args.prefix}_embeddings.npy"
    ids_path = out_dir / f"{args.prefix}_image_ids.npy"
    meta_path = out_dir / f"{args.prefix}_metadata.json"

    metadata = {
        "model": "vit_b_16",
        "weights": "ViT_B_16_Weights.DEFAULT",
        "dimension": DIMENSION,
        "count": total,
        "normalized": True,
        "order": "sorted by filename",
        "source_type": source_type,
        "source_path": source_path,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "rows_done": 0,
    }

    # Resume only when the previous run used the same source and size.
    start_row = 0
    if emb_path.exists() and ids_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if (
            old.get("count") == total
            and old.get("source_path") == source_path
            and old.get("dimension") == DIMENSION
        ):
            start_row = old.get("rows_done", 0)
            print(f"resuming from row {start_row}/{total}", flush=True)

    mode = "r+" if start_row > 0 else "w+"
    embeddings = np.lib.format.open_memmap(
        emb_path, mode=mode, dtype="float32", shape=(total, DIMENSION)
    )
    id_array = np.lib.format.open_memmap(
        ids_path, mode=mode, dtype="int64", shape=(total,)
    )
    id_array[start_row:] = ids[start_row:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    # Same modification as the notebook: drop the classifier head so the
    # model returns the 768-d image embedding.
    model.heads = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()

    def prepare(chunk):
        tensors = [
            preprocess(load_image(name, args.zip).convert("RGB"))
            for name in chunk
        ]
        return torch.stack(tensors)

    # Prefetch the next batch on CPU threads while the GPU computes.
    pool = ThreadPoolExecutor(max_workers=2)
    batch_starts = list(range(start_row, total, args.batch_size))
    iterator = iter(batch_starts)

    first = next(iterator, None)
    pending = pool.submit(prepare, names[first : first + args.batch_size]) if first is not None else None

    started = time.time()
    rows_done = start_row
    with torch.inference_mode():
        while pending is not None:
            start_index = first
            batch = pending.result()

            nxt = next(iterator, None)
            pending = (
                pool.submit(prepare, names[nxt : nxt + args.batch_size])
                if nxt is not None
                else None
            )

            images = batch.to(device)
            if args.amp:
                with torch.autocast(device_type=device.type):
                    features = model(images)
            else:
                features = model(images)

            # fp32 + L2 normalize so IndexFlatIP == cosine similarity.
            features = F.normalize(features.float(), p=2, dim=1).cpu().numpy()
            embeddings[start_index : start_index + features.shape[0]] = features
            rows_done = start_index + features.shape[0]

            metadata["rows_done"] = rows_done
            meta_path.write_text(json.dumps(metadata, indent=2))

            processed = rows_done - start_row
            elapsed = max(time.time() - started, 1e-9)
            rate = processed / elapsed
            eta = (total - rows_done) / max(rate, 1e-9) / 60
            print(
                f"{rows_done}/{total} ({rate:.1f} img/s, eta {eta:.0f} min)",
                flush=True,
            )
            first = nxt

    embeddings.flush()
    id_array.flush()
    metadata["rows_done"] = rows_done
    meta_path.write_text(json.dumps(metadata, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
