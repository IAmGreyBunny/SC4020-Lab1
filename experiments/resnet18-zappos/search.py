"""Run one exact cosine search using the ResNet18 `.npy` handoff files."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from common import IMAGE_ROOT, prepare_data


# Generate a PNG without opening a blocking GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/zappos/resnet_embeddings.npy")
    parser.add_argument("--ids", default="data/zappos/resnet_image_ids.npy")
    parser.add_argument("--images-dir", default=str(IMAGE_ROOT))
    parser.add_argument(
        "--out-dir",
        default="experiments/resnet18-zappos/results",
    )
    parser.add_argument("--query-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", default="5,10,20")
    return parser.parse_args()


def metrics_at_k(query_row, result_rows, labels):
    query_label = labels[query_row]
    relevant_retrieved = int(np.count_nonzero(labels[result_rows] == query_label))
    relevant_available = int(np.count_nonzero(labels == query_label) - 1)
    precision = relevant_retrieved / len(result_rows)
    recall = relevant_retrieved / relevant_available if relevant_available else 0.0
    return precision, recall, relevant_retrieved, relevant_available


def save_result_grid(output_path, query_row, result_rows, scores, dataframe):
    total_images = len(result_rows) + 1
    columns = 5
    rows = int(np.ceil(total_images / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3.5))
    axes = np.atleast_1d(axes).ravel()

    query = dataframe.iloc[query_row]
    with Image.open(query["image"]) as image:
        axes[0].imshow(image.convert("RGB"))
    axes[0].set_title(
        f"Query ID {query_row}\n{query['category']} / {query['subcategory']}"
    )
    axes[0].axis("off")

    for rank, (result_row, score) in enumerate(zip(result_rows, scores), start=1):
        result = dataframe.iloc[result_row]
        with Image.open(result["image"]) as image:
            axes[rank].imshow(image.convert("RGB"))
        axes[rank].set_title(
            f"Rank {rank}\n{result['category']} / {result['subcategory']}"
            f"\nScore: {score:.3f}"
        )
        axes[rank].axis("off")

    for axis in axes[total_images:]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    k_values = tuple(sorted(set(int(value) for value in args.ks.split(","))))
    embeddings = np.load(args.emb, mmap_mode="r")
    image_ids = np.load(args.ids, mmap_mode="r")
    dataframe = prepare_data(image_root=args.images_dir)

    if embeddings.shape != (len(image_ids), 512):
        raise RuntimeError(
            f"Expected ({len(image_ids)}, 512) embeddings, got {embeddings.shape}"
        )
    if len(dataframe) != len(image_ids):
        raise RuntimeError(
            "Sorted image count does not match embedding/image-ID row count"
        )
    if not np.array_equal(image_ids, np.arange(len(image_ids), dtype=np.int64)):
        raise RuntimeError("Image IDs do not match deterministic sorted row IDs")

    if args.query_id is None:
        # Reproducible random nonzero ID, so this does not reuse query 0.
        rng = np.random.default_rng(args.seed)
        query_id = int(rng.integers(1, len(image_ids)))
    else:
        query_id = args.query_id

    matching_rows = np.flatnonzero(image_ids == query_id)
    if len(matching_rows) != 1:
        raise RuntimeError(f"Expected exactly one row for query ID {query_id}")
    query_row = int(matching_rows[0])

    scores = embeddings @ embeddings[query_row]
    scores[query_row] = -np.inf
    max_k = max(k_values)
    result_rows = np.argsort(scores)[::-1][:max_k]
    result_scores = scores[result_rows]

    categories = dataframe["category"].to_numpy()
    subcategories = dataframe["subcategory"].to_numpy()
    query = dataframe.iloc[query_row]
    summary_rows = []

    print(f"Database: {len(embeddings)} images")
    print(f"Query ID: {query_id}")
    print(f"Query image: {query['image']}")
    print(f"Query label: {query['category']} / {query['subcategory']}")
    print(f"\nTop {max_k} results:")

    for rank, (row, score) in enumerate(zip(result_rows, result_scores), start=1):
        result = dataframe.iloc[row]
        print(
            f"{rank}. ID {int(image_ids[row])} | similarity={score:.4f} | "
            f"{result['category']} / {result['subcategory']} | {result['image']}"
        )

    for k in k_values:
        top_rows = result_rows[:k]
        category_precision, category_recall, category_hits, category_total = metrics_at_k(
            query_row, top_rows, categories
        )
        subcategory_precision, subcategory_recall, subcategory_hits, subcategory_total = metrics_at_k(
            query_row, top_rows, subcategories
        )
        summary_rows.append(
            {
                "dataset_size": len(embeddings),
                "query_id": query_id,
                "query_row": query_row,
                "query_category": query["category"],
                "query_subcategory": query["subcategory"],
                "k": k,
                "category_hits": category_hits,
                "category_relevant_available": category_total,
                "category_precision": category_precision,
                "category_recall": category_recall,
                "subcategory_hits": subcategory_hits,
                "subcategory_relevant_available": subcategory_total,
                "subcategory_precision": subcategory_precision,
                "subcategory_recall": subcategory_recall,
            }
        )

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / (
        f"single_query_summary_N{len(embeddings)}_Q{query_id}.csv"
    )
    image_path = output_directory / (
        f"single_query_top{max_k}_N{len(embeddings)}_Q{query_id}.png"
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    save_result_grid(
        image_path,
        query_row,
        result_rows,
        result_scores,
        dataframe,
    )

    print("\nSingle-query summary:")
    print(
        summary[
            [
                "k",
                "category_precision",
                "category_recall",
                "subcategory_precision",
                "subcategory_recall",
            ]
        ].to_string(index=False)
    )
    print(f"\nSummary saved to: {summary_path}")
    print(f"Result image saved to: {image_path}")


if __name__ == "__main__":
    main()
