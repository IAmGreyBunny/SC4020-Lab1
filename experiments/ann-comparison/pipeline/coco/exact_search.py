"""FAISS exact-search baseline on generated ViT embeddings.

New file for this experiment. Uses FAISS IndexFlatIP (brute-force inner
product). Because the embeddings are L2-normalized, inner product equals
cosine similarity. This produces:

1. The exact top-K neighbours for a fixed set of query images. These are
   saved as ground truth for later ANN recall (HNSW, IVF-PQ).
2. Timing metrics (latency, QPS) and index statistics.
3. A readable predictions JSON of image ids per query.

The query image itself is excluded from its own result list.
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np


def drop_self(row, self_index, k):
    # self appears with similarity 1.0; remove it, then keep top-k.
    kept = [index for index in row if index != self_index]
    return kept[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)

    embeddings = np.ascontiguousarray(np.load(args.emb))
    ids = np.load(args.ids)
    count, dimension = embeddings.shape

    norms = np.linalg.norm(embeddings[:1024], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("note: vectors not normalized - normalizing in memory")
        faiss.normalize_L2(embeddings)

    if args.num_queries > count:
        raise SystemExit("num-queries cannot exceed dataset size")

    # Index build timing.
    start = time.perf_counter()
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    build_seconds = time.perf_counter() - start

    # Fixed query set for all future ANN comparisons.
    rng = np.random.default_rng(args.seed)
    query_rows = rng.choice(count, size=args.num_queries, replace=False)
    queries = np.ascontiguousarray(embeddings[query_rows])

    # Warm-up, then repeated timed batch searches (median for stability).
    index.search(queries[: min(10, len(queries))], args.k + 1)
    timings = []
    neighbours = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        _, neighbours = index.search(queries, args.k + 1)
        timings.append(time.perf_counter() - start)
    search_seconds = float(np.median(timings))

    # Correctness cross-check: FAISS vs plain NumPy on a few queries.
    for row in query_rows[:5]:
        scores = embeddings @ embeddings[row]
        expected = np.argsort(scores)[::-1][: args.k + 1]
        assert row in expected, "self missing from numpy top-k"
        expected = [int(i) for i in expected if i != row][: args.k]
        actual = drop_self(list(neighbours[list(query_rows).index(row)]), row, args.k)
        assert actual[0] == expected[0], f"top-1 mismatch at row {row}"

    top_k_rows = []
    predictions = {}
    for position, query_row in enumerate(query_rows):
        kept = drop_self([int(i) for i in neighbours[position]], int(query_row), args.k)
        top_k_rows.append(kept)
        predictions[str(int(ids[query_row]))] = [int(ids[i]) for i in kept]

    top_k_rows = np.array(top_k_rows, dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ground_truth_rows.npy", top_k_rows)
    np.save(out_dir / "query_rows.npy", query_rows)

    summary = {
        "algorithm": "Exact FlatIP",
        "dataset_size": count,
        "dimension": dimension,
        "num_queries": args.num_queries,
        "k": args.k,
        "seed": args.seed,
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "latency_ms_per_query": search_seconds * 1000 / args.num_queries,
        "queries_per_second": args.num_queries / search_seconds,
        "index_size_mb": len(faiss.serialize_index(index)) / (1024 * 1024),
        "faiss_threads": args.threads,
    }
    with (out_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    with (out_dir / "predictions.json").open("w") as file:
        json.dump(predictions, file, indent=2)

    print(
        f"dataset={count} dim={dimension} queries={args.num_queries} k={args.k}\n"
        f"build={build_seconds:.3f}s  latency={summary['latency_ms_per_query']:.3f} ms/query  "
        f"qps={summary['queries_per_second']:.1f}  index={summary['index_size_mb']:.1f} MB"
    )
    print(f"ground truth saved to {out_dir / 'ground_truth_rows.npy'}")


if __name__ == "__main__":
    main()
