"""Evaluate the exact ResNet18 `.npy` artifacts used for team handoff."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from common import IMAGE_ROOT, prepare_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/zappos/resnet_embeddings.npy")
    parser.add_argument("--ids", default="data/zappos/resnet_image_ids.npy")
    parser.add_argument("--images-dir", default=str(IMAGE_ROOT))
    parser.add_argument(
        "--out-dir",
        default="experiments/resnet18-zappos/results",
    )
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metrics_at_k(query_index, result_indices, labels):
    query_label = labels[query_index]
    relevant_retrieved = np.count_nonzero(labels[result_indices] == query_label)
    relevant_available = np.count_nonzero(labels == query_label) - 1
    precision = relevant_retrieved / len(result_indices)
    recall = relevant_retrieved / relevant_available if relevant_available else 0.0
    return precision, recall


def main():
    args = parse_args()
    k_values = tuple(int(value) for value in args.ks.split(","))
    embeddings = np.load(args.emb, mmap_mode="r")
    image_ids = np.load(args.ids, mmap_mode="r")
    dataframe = prepare_data(image_root=args.images_dir)

    expected_ids = np.arange(len(image_ids), dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise RuntimeError(f"Expected embeddings with shape (N, 512), got {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise RuntimeError(f"Expected float32 embeddings, got {embeddings.dtype}")
    if image_ids.dtype != np.int64:
        raise RuntimeError(f"Expected int64 image IDs, got {image_ids.dtype}")
    if len(embeddings) != len(image_ids) or len(image_ids) != len(dataframe):
        raise RuntimeError(
            "Embedding, image-ID, and sorted image-path row counts do not match: "
            f"{len(embeddings)}, {len(image_ids)}, {len(dataframe)}"
        )
    if not np.array_equal(image_ids, expected_ids):
        raise RuntimeError("Image IDs do not match deterministic sorted row IDs")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embeddings are not L2-normalized")

    categories = dataframe["category"].to_numpy()
    subcategories = dataframe["subcategory"].to_numpy()
    query_count = min(args.num_queries, len(embeddings))
    rng = np.random.default_rng(args.seed)
    query_indices = rng.choice(len(embeddings), size=query_count, replace=False)
    rows = []

    for number, query_index in enumerate(query_indices, start=1):
        start = perf_counter()
        scores = embeddings @ embeddings[query_index]
        scores[query_index] = -np.inf
        ranked_indices = np.argsort(scores)[::-1][:max(k_values)]
        query_time_ms = (perf_counter() - start) * 1000

        for k in k_values:
            result_indices = ranked_indices[:k]
            category_precision, category_recall = metrics_at_k(
                query_index, result_indices, categories
            )
            subcategory_precision, subcategory_recall = metrics_at_k(
                query_index, result_indices, subcategories
            )
            rows.append(
                {
                    "dataset_size": len(embeddings),
                    "query_row": int(query_index),
                    "query_image_id": int(image_ids[query_index]),
                    "k": k,
                    "query_category": categories[query_index],
                    "query_subcategory": subcategories[query_index],
                    "category_precision": category_precision,
                    "category_recall": category_recall,
                    "subcategory_precision": subcategory_precision,
                    "subcategory_recall": subcategory_recall,
                    "query_time_ms": query_time_ms,
                }
            )

        if number % 50 == 0 or number == query_count:
            print(f"Evaluated query {number}/{query_count}", flush=True)

    details = pd.DataFrame(rows)
    summary = (
        details.groupby("k", as_index=False)
        .agg(
            category_precision=("category_precision", "mean"),
            category_recall=("category_recall", "mean"),
            subcategory_precision=("subcategory_precision", "mean"),
            subcategory_recall=("subcategory_recall", "mean"),
            average_query_time_ms=("query_time_ms", "mean"),
        )
    )

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    suffix = f"resnet_npy_N{len(embeddings)}"
    details_path = output_directory / f"evaluation_details_{suffix}.csv"
    summary_path = output_directory / f"evaluation_summary_{suffix}.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nValidated exact handoff files: {args.emb} and {args.ids}")
    print(f"Average results over {query_count} queries:")
    print(summary.to_string(index=False))
    print(f"\nDetailed results saved to: {details_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
