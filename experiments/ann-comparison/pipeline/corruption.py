"""Corrupted-query robustness for the 2x2 model x dataset grid.

Measures what happens to *retrieval* when the query image is corrupted while
the gallery stays clean. This is stronger than embedding drift alone: two
embeddings can move (cosine < 1) without changing the neighbour ranking, and
two can reorder without moving much. Both are reported here.

Protocol (identical for every configuration):
- Queries: the same fixed seed-42 rows that benchmark_ann.py holds out,
  reduced to a reproducible subset (default 200).
- Gallery: every non-query row, clean.
- Corruption: only the query, swept over TRANSFORMS x levels
  (pipeline/perturbations.py). The clean query is included as the
  `original` level, which must score cosine = recall = mAP = 1.0.
- Search: exact FlatIP (isolates the model's effect from ANN error).
- Metrics vs the clean query's exact top-k ground truth:
    cosine(clean, corrupted)      embedding drift
    Recall@K / Precision@K        rank stability
    mAP@K                         rank-aware stability

Clean query embeddings are read from data/<dir>/<model>_embeddings.npy; only
the corrupted variants are embedded (and cached), so nothing is recomputed.

Outputs per configuration in results/<config>/robustness/:
- corruption_per_query.csv         one row per query/variant
- corruption_summary.csv           mean/std per transform/level
- corruption_params.json           models, transforms, seed, query rows
- corrupted_query_embeddings.npy   cached variant vectors (git-ignored)
- corrupted_query_variants.csv     variant metadata matching the npy
- corrupted_query_top20.npy        top-k gallery rows per variant (for the notebook grid)
- corrupted_query_clean_top20.npy  top-k gallery rows of the clean query per query

And results/robustness_corruption_combined.csv across all configurations.

Example (from the repository root):
    python experiments/ann-comparison/pipeline/corruption.py --config all --num-queries 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PIPELINE_DIR.parent
# datasets.py, benchmark_ann.py live in the experiment folder; models.py and
# perturbations.py next to this file.
for _path in (str(EXPERIMENT_DIR), str(PIPELINE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import faiss
import pandas as pd
from PIL import Image

from benchmark_ann import map_at_k, precision_at_k, recall_at_k
from datasets import CONFIGS, RESULTS_DIR, results_dir
from models import create_model, embed_images, list_images
from perturbations import TRANSFORMS, level_label, perturb

BENCHMARK_QUERIES = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--num-queries", type=int, default=200,
                        help="subset of the benchmark queries to corrupt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-gt", type=int, default=20)
    parser.add_argument("--metric-ks", default="5,10,20")
    parser.add_argument("--transforms", default=None,
                        help="comma-separated subset of transform names")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--emb", help="override the registry path (single config only)")
    parser.add_argument("--ids", help="override the registry path (single config only)")
    parser.add_argument("--images-dir", help="override the registry image folder (single config only)")
    return parser.parse_args()


def select_transforms(requested):
    valid = {spec["name"]: spec for spec in TRANSFORMS}
    if requested is None:
        return list(TRANSFORMS)
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = set(names) - set(valid)
    if unknown:
        raise SystemExit(f"unknown transform(s): {sorted(unknown)}")
    return [spec for spec in TRANSFORMS if spec["name"] in names]


def select_query_rows(key, count, args, per_query_dir):
    """Return (query_rows, gallery_rows) for this run.

    Prefer the benchmark's saved split; otherwise reproduce it exactly from
    the run's recorded seed / query count. The corrupted run then takes a
    reproducible subset of those queries so it stays comparable to the ANN
    benchmark while staying cheap.
    """
    meta_path = results_dir(key) / "ann_results.json"
    bench_seed, bench_queries = args.seed, BENCHMARK_QUERIES
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())["metadata"]
        bench_seed = metadata.get("seed", bench_seed)
        bench_queries = metadata.get("num_queries", bench_queries)

    query_path = per_query_dir / "query_rows.npy"
    gallery_path = per_query_dir / "gallery_rows.npy"
    if query_path.exists() and gallery_path.exists():
        full_queries = np.load(query_path)
        gallery_rows = np.load(gallery_path)
    else:
        permutation = np.random.default_rng(bench_seed).permutation(count)
        full_queries = np.sort(permutation[: bench_queries])
        gallery_rows = np.sort(permutation[bench_queries :])

    subset_size = min(args.num_queries, len(full_queries))
    if subset_size < len(full_queries):
        picked = np.random.default_rng(args.seed + 99).choice(
            len(full_queries), size=subset_size, replace=False
        )
        query_rows = np.sort(full_queries[np.sort(picked)])
    else:
        query_rows = np.sort(full_queries)
    return query_rows, gallery_rows


def variant_plan(selected):
    """The (transform, value, label) triple for every corrupted level.

    The clean query is the `original` level (empty value); it anchors the
    sanity check and appears in the summaries as its own row.
    """
    plan = [("original", "", "-")]
    for spec in selected:
        for value in spec["values"]:
            plan.append((spec["name"], value, level_label(spec, value)))
    return plan


def embed_corrupted(paths, query_rows, selected, model, preprocess, device, batch_size):
    """Embed original + corrupted variants of each query, streaming to bound memory.

    Returns (vectors (N, d), meta list of dicts aligned with vectors).
    """
    plan = variant_plan(selected)
    buffer_images, buffer_meta, chunks = [], [], []

    def flush():
        if buffer_images:
            chunks.append(embed_images(buffer_images, model, preprocess, device, batch_size))
            buffer_images.clear()

    for position, row in enumerate(query_rows):
        with Image.open(paths[row]) as handle:
            image = handle.convert("RGB")
        for transform, value, label in plan:
            variant = image if transform == "original" else perturb(image, transform, value)
            buffer_images.append(variant)
            buffer_meta.append({
                "query": position, "transform": transform,
                "level": value, "label": label,
            })
        if len(buffer_images) >= batch_size:
            flush()
    flush()

    vectors = np.concatenate(chunks, axis=0).astype(np.float32)
    if vectors.shape[0] != len(buffer_meta):
        raise RuntimeError("variant vectors and metadata are misaligned")
    return vectors, buffer_meta


def summarize(frame, metric_ks):
    metrics = ["cosine", *[f"recall_at_{k}" for k in metric_ks],
               *[f"precision_at_{k}" for k in metric_ks], *[f"map_at_{k}" for k in metric_ks]]
    summary = (
        frame.groupby(["model", "dataset", "transform", "level"])[metrics]
        .agg(["mean", "std"])
        .round(6)
    )
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary.reset_index()


def run_config(key, cfg, args, selected):
    import torch

    emb_path = Path(args.emb) if args.emb else cfg["emb"]
    ids_path = Path(args.ids) if args.ids else cfg["ids"]
    image_dir = Path(args.images_dir) if args.images_dir else cfg["image_dir"]

    if not emb_path.exists() or not ids_path.exists():
        print(f"[{key}] embeddings not found: {emb_path} - skipping", flush=True)
        return None
    if not image_dir.exists():
        print(f"[{key}] images not found: {image_dir} - skipping", flush=True)
        return None

    embedding = np.load(emb_path)
    ids = np.load(ids_path)
    count, dimension = embedding.shape
    if len(ids) != count:
        raise SystemExit(f"[{key}] id count does not match embedding rows")

    paths, _ = list_images(cfg["dataset"], image_dir)
    if len(paths) != count:
        raise SystemExit(
            f"[{key}] found {len(paths)} images under {image_dir} but {count} embeddings.\n"
            "The image folder must be the one the embeddings were extracted from."
        )

    per_query_dir = results_dir(key) / "per_query"
    query_rows, gallery_rows = select_query_rows(key, count, args, per_query_dir)
    print(f"[{key}] {len(query_rows)} queries, {len(gallery_rows):,} gallery, {dimension}-d",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]
    model, preprocess, _ = create_model(model_name)
    model.to(device)
    batch_size = args.batch_size or (64 if model_name == "vit" else 128)

    vectors, meta = embed_corrupted(
        paths, query_rows, selected, model, preprocess, device, batch_size
    )
    clean = np.ascontiguousarray(embedding[query_rows])
    gallery = np.ascontiguousarray(embedding[gallery_rows])
    del model

    index = faiss.IndexFlatIP(dimension)
    index.add(gallery)
    _, gt_indices = index.search(clean, args.k_gt)
    _, variant_indices = index.search(vectors, args.k_gt)

    query_of_variant = np.array([variant["query"] for variant in meta])
    gt_rows = gt_indices[query_of_variant]
    cosines = np.einsum("ij,ij->i", clean[query_of_variant], vectors)

    rows = []
    for position, variant in enumerate(meta):
        row = {
            "config": key,
            "model": model_name,
            "dataset": cfg["dataset"],
            "query": variant["query"],
            "image_id": int(ids[query_rows[variant["query"]]]),
            "transform": variant["transform"],
            "level": variant["level"],
            "label": variant["label"],
            "cosine": round(float(cosines[position]), 6),
        }
        for k in args.metric_ks:
            row[f"recall_at_{k}"] = round(recall_at_k(gt_rows[position:position + 1],
                                                      variant_indices[position:position + 1], k), 6)
            row[f"precision_at_{k}"] = round(precision_at_k(gt_rows[position:position + 1],
                                                            variant_indices[position:position + 1], k), 6)
            row[f"map_at_{k}"] = round(map_at_k(gt_rows[position:position + 1],
                                                variant_indices[position:position + 1], k), 6)
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = summarize(frame, args.metric_ks)

    out_dir = results_dir(key) / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "corruption_per_query.csv", index=False)
    summary.to_csv(out_dir / "corruption_summary.csv", index=False)
    np.save(out_dir / "corrupted_query_embeddings.npy", vectors)
    np.save(out_dir / "corrupted_query_top20.npy", variant_indices)
    np.save(out_dir / "corrupted_query_clean_top20.npy", gt_indices)
    pd.DataFrame(meta).assign(image_id=[int(ids[query_rows[m["query"]]]) for m in meta]).to_csv(
        out_dir / "corrupted_query_variants.csv", index=False
    )
    params = {
        "config": key,
        "model": model_name,
        "dataset": cfg["dataset"],
        "dimension": dimension,
        "num_queries": int(len(query_rows)),
        "gallery_size": int(len(gallery_rows)),
        "seed": args.seed,
        "k_gt": args.k_gt,
        "metric_ks": list(args.metric_ks),
        "transforms": [spec["name"] for spec in selected],
        "device": str(device),
        "query_rows": query_rows.tolist(),
    }
    (out_dir / "corruption_params.json").write_text(json.dumps(params, indent=2))

    original = frame[frame["transform"] == "original"]
    drift = original["cosine"].min()
    recall10 = original["recall_at_10"].min() if 10 in args.metric_ks else float("nan")
    print(f"[{key}] original self-check: cosine {drift:.4f}, recall@10 {recall10:.4f}", flush=True)
    print(f"[{key}] results in {out_dir}", flush=True)
    return summary.assign(config=key)


def write_combined(summaries):
    combined = [frame for frame in summaries if frame is not None]
    if not combined:
        return
    out_path = RESULTS_DIR / "robustness_corruption_combined.csv"
    pd.concat(combined, ignore_index=True).to_csv(out_path, index=False)
    print(f"combined table: {out_path}", flush=True)


def main():
    args = parse_args()
    args.metric_ks = [int(x) for x in args.metric_ks.split(",")]
    selected = select_transforms(args.transforms)

    keys = list(CONFIGS) if args.config == "all" else [args.config]
    if len(keys) > 1 and (args.emb or args.ids or args.images_dir):
        raise SystemExit("--emb/--ids/--images-dir need a single --config")

    print(f"transforms: {[spec['name'] for spec in selected]}")
    summaries = []
    for key in keys:
        summaries.append(run_config(key, CONFIGS[key], args, selected))
    write_combined(summaries)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
