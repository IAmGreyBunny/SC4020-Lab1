"""Generate image embeddings for any model x dataset combination.

One extractor for the 2x2 comparison: {ViT-B/16, ResNet18} x {COCO,
UT-Zap50K}. It merges the two per-dataset extractors in pipeline/coco and
pipeline/zappos and keeps their conventions, so outputs are drop-in
compatible with the files the team already shares:

- models   : torchvision pretrained weights (DEFAULT), classifier removed
             (ViT heads / ResNet fc -> Identity), weights.transforms()
- ordering : COCO sorted by filename, id = numeric stem;
             Zappos sorted by full path, id = row number 0..N-1
- output   : data/<circo|zappos>/<model>_embeddings.npy  float32 (N, d),
             L2-normalized; <model>_image_ids.npy int64 (N,);
             <model>_metadata.json with progress (rows_done)

Batched GPU inference with CPU prefetch, and resumable: re-running the
same command continues from the last completed row.

Examples (from the repository root):
    python experiments/ann-comparison/pipeline/extract_embeddings.py --model vit --dataset coco
    python experiments/ann-comparison/pipeline/extract_embeddings.py --model resnet --dataset zappos
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Models, image ordering and the dataset registry live in models.py so this
# extractor and corruption.py share a single definition; see models.py.
from models import DATASETS, MODELS, create_model, list_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--images-dir", help="override the default image folder")
    parser.add_argument("--output-dir", help="override the default output folder")
    parser.add_argument("--prefix", help="output prefix (default: the model name)")
    parser.add_argument("--limit", type=int, help="only the first N images (smoke tests)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4, help="image decoding threads")
    args = parser.parse_args()

    spec = MODELS[args.model]
    dimension = spec["dimension"]
    batch_size = args.batch_size or spec["batch_size"]
    images_dir = Path(args.images_dir) if args.images_dir else DATASETS[args.dataset]["images_dir"]
    out_dir = Path(args.output_dir) if args.output_dir else DATASETS[args.dataset]["output_dir"]
    prefix = args.prefix or args.model

    if not images_dir.exists():
        raise SystemExit(f"image folder not found: {images_dir} (see README.md)")
    paths, ids = list_images(args.dataset, images_dir)
    if args.limit is not None:
        paths, ids = paths[: args.limit], ids[: args.limit]
    total = len(paths)
    if total == 0:
        raise SystemExit(f"no JPG images found under {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{prefix}_embeddings.npy"
    ids_path = out_dir / f"{prefix}_image_ids.npy"
    meta_path = out_dir / f"{prefix}_metadata.json"

    model, preprocess, weights_name = create_model(args.model)
    metadata = {
        "model": args.model,
        "weights": weights_name,
        "dataset": args.dataset,
        "dimension": dimension,
        "count": total,
        "normalized": True,
        "order": "sorted by filename" if args.dataset == "coco" else "sorted by full path",
        "source_path": str(images_dir),
        "batch_size": batch_size,
        "rows_done": 0,
    }

    # Resume only when the previous run used the same source, model and size.
    start_row = 0
    if emb_path.exists() and ids_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if (
            old.get("count") == total
            and old.get("source_path") == str(images_dir)
            and old.get("model") == args.model
            and old.get("dimension") == dimension
        ):
            start_row = old.get("rows_done", 0)
            print(f"resuming from row {start_row}/{total}", flush=True)
    if start_row >= total:
        print("already complete", flush=True)
        return

    mode = "r+" if start_row > 0 else "w+"
    try:
        embeddings = np.lib.format.open_memmap(emb_path, mode=mode, dtype="float32", shape=(total, dimension))
        id_array = np.lib.format.open_memmap(ids_path, mode=mode, dtype="int64", shape=(total,))
    except OSError as error:
        # Windows refuses to rewrite a file that another process has
        # memory-mapped, e.g. a Jupyter kernel that ran np.load(..., mmap_mode="r").
        raise SystemExit(
            f"cannot write {emb_path.name} ({error}).\n"
            "Another process probably has the old file open - usually a running "
            "Jupyter kernel that loaded it. Shut that kernel down and re-run."
        )
    id_array[:] = ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"{args.model} x {args.dataset}: {total} images, device {device}, batch {batch_size}", flush=True)

    def load(path):
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    decoder = ThreadPoolExecutor(max_workers=args.workers)

    def prepare(start):
        return torch.stack(list(decoder.map(load, paths[start : start + batch_size])))

    # Prefetch the next batch on CPU threads while the GPU computes.
    prefetch = ThreadPoolExecutor(max_workers=1)
    starts = list(range(start_row, total, batch_size))
    pending = prefetch.submit(prepare, starts[0])

    started = time.time()
    rows_done = start_row
    with torch.inference_mode():
        for i, start in enumerate(starts):
            batch = pending.result()
            if i + 1 < len(starts):
                pending = prefetch.submit(prepare, starts[i + 1])

            # fp32 + L2 normalize so IndexFlatIP == cosine similarity.
            features = F.normalize(model(batch.to(device)).float(), p=2, dim=1).cpu().numpy()
            embeddings[start : start + len(features)] = features
            rows_done = start + len(features)

            metadata["rows_done"] = rows_done
            meta_path.write_text(json.dumps(metadata, indent=2))
            if i % 20 == 0 or rows_done == total:
                rate = (rows_done - start_row) / max(time.time() - started, 1e-9)
                eta = (total - rows_done) / max(rate, 1e-9) / 60
                print(f"{rows_done}/{total} ({rate:.0f} img/s, eta {eta:.1f} min)", flush=True)

    embeddings.flush()
    id_array.flush()
    del embeddings, id_array

    # Reload and check the output contract.
    saved = np.load(emb_path, mmap_mode="r")
    norms = np.linalg.norm(saved[: min(total, 4096)], axis=1)
    assert saved.shape == (total, dimension) and saved.dtype == np.float32
    assert np.isfinite(saved[: min(total, 4096)]).all()
    assert np.allclose(norms, 1.0, atol=1e-3)
    print(f"DONE {emb_path} {saved.shape} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
