from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from common import IMAGE_ROOT


EMBEDDINGS_PATH = Path(
    "output/zappos_embeddings_N50025.pkl"
)

K_VALUES = (5, 10, 20)
NUMBER_OF_QUERIES = 500
RANDOM_SEED = 42


def get_subcategory(path):
    return Path(path).relative_to(IMAGE_ROOT).parts[1]


def metrics_at_k(query_index, result_indices, labels):
    query_label = labels[query_index]
    relevant_retrieved = sum(labels[index] == query_label for index in result_indices)
    relevant_available = sum(label == query_label for label in labels) - 1

    precision = relevant_retrieved / len(result_indices)
    recall = relevant_retrieved / relevant_available if relevant_available else 0.0
    return precision, recall


def main():
    data = joblib.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    paths = data["paths"]
    categories = data["categories"]
    subcategories = [get_subcategory(path) for path in paths]

    rng = np.random.default_rng(RANDOM_SEED)
    query_count = min(NUMBER_OF_QUERIES, len(paths))
    query_indices = rng.choice(len(paths), size=query_count, replace=False)
    rows = []

    for number, query_index in enumerate(query_indices, start=1):
        start = perf_counter()
        scores = embeddings @ embeddings[query_index]
        scores[query_index] = -np.inf
        ranked_indices = np.argsort(scores)[::-1][:max(K_VALUES)]
        query_time_ms = (perf_counter() - start) * 1000

        for k in K_VALUES:
            result_indices = ranked_indices[:k]
            category_precision, category_recall = metrics_at_k(
                query_index, result_indices, categories
            )
            subcategory_precision, subcategory_recall = metrics_at_k(
                query_index, result_indices, subcategories
            )
            rows.append({
                "dataset_size": len(paths),
                "query_index": int(query_index),
                "k": k,
                "query_category": categories[query_index],
                "query_subcategory": subcategories[query_index],
                "category_precision": category_precision,
                "category_recall": category_recall,
                "subcategory_precision": subcategory_precision,
                "subcategory_recall": subcategory_recall,
                "query_time_ms": query_time_ms,
            })

        print(f"Evaluated query {number}/{query_count}")

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

    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)
    details_path = output_directory / f"evaluation_details_N{len(paths)}.csv"
    summary_path = output_directory / f"evaluation_summary_N{len(paths)}.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nAverage results over", query_count, "queries:")
    print(summary.to_string(index=False))
    print("\nDetailed results saved to:", details_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
