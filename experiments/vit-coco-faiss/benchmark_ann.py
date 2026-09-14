"""Benchmark ANN search methods on the real ViT COCO embeddings.

New file for this experiment. Compares, on identical data:

- Exact FlatIP : brute-force baseline, produces the ground truth
- HNSW         : graph-based ANN, efSearch sweep
- IVF-PQ       : clustered + compressed ANN, nprobe sweep
- LSH          : binary hashing (FAISS IndexLSH), nbits sweep

Setup: 1,000 query images are held out (seed 42); the remaining 122,403
vectors form the gallery. IVF-PQ is trained only on gallery vectors. All
vectors are already L2-normalized, so inner product equals cosine
similarity. On normalized vectors L2 ranking equals cosine ranking, so
LSH (L2 by default in FAISS) is comparable to the IP-based methods.

Outputs (in --out-dir):
- ann_results.json / ann_results.csv      aggregate metrics per setting
- per_query/exact_top20.npy               exact top-20 gallery rows
- per_query/<setting>_top20.npy           top-20 for every ANN setting
- per_query/query_rows.npy                the 1,000 fixed query rows
- per_query/gallery_rows.npy              gallery row -> original row
"""

import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np

RECALL_KS = (5, 10, 20)


def timed_search(index, queries, k, repeats):
    # Warm up, then take the median of repeated batch searches.
    index.search(queries[: min(10, len(queries))], k)
    timings = []
    indices = None
    for _ in range(repeats):
        start = time.perf_counter()
        _, indices = index.search(queries, k)
        timings.append(time.perf_counter() - start)
    return indices, float(np.median(timings))


def recall_at_k(ground_truth, actual, k):
    # Fraction of the exact top-k that the ANN method also returns.
    # -1 entries (missing results) never match valid gallery rows.
    recalls = []
    for gt_row, actual_row in zip(ground_truth, actual):
        gt_set = set(gt_row[:k].tolist())
        actual_set = set(actual_row[:k].tolist())
        recalls.append(len(gt_set & actual_set) / k)
    return float(np.mean(recalls))


def result_row(algorithm, parameters, build_seconds, search_seconds,
               num_queries, index, ground_truth, indices):
    row = {
        "algorithm": algorithm,
        "parameters": parameters,
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "latency_ms_per_query": search_seconds * 1000 / num_queries,
        "queries_per_second": num_queries / search_seconds,
        "index_size_mb": len(faiss.serialize_index(index)) / (1024 * 1024),
    }
    for k in RECALL_KS:
        row[f"recall_at_{k}"] = recall_at_k(ground_truth, indices, k)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/circo/vit_embeddings.npy")
    parser.add_argument("--ids", default="data/circo/vit_image_ids.npy")
    parser.add_argument("--out-dir", default="experiments/vit-coco-faiss/results/ann")
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-gt", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=0)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", default="16,32,64,128,256")
    parser.add_argument("--ivf-nlist", type=int, default=256)
    parser.add_argument("--ivf-nprobe", default="1,4,8,16,32,64")
    parser.add_argument("--pq-m", type=int, default=48)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--ivf-train-sample", type=int, default=50000)
    parser.add_argument("--lsh-nbits", default="256,512")
    args = parser.parse_args()

    hnsw_ef_list = [int(x) for x in args.hnsw_ef_search.split(",")]
    nprobe_list = [int(x) for x in args.ivf_nprobe.split(",")]
    lsh_nbits_list = [int(x) for x in args.lsh_nbits.split(",")]

    embeddings = np.ascontiguousarray(np.load(args.emb))
    ids = np.load(args.ids)
    count, dimension = embeddings.shape

    norms = np.linalg.norm(embeddings[:1024], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("note: vectors not normalized - normalizing in memory")
        faiss.normalize_L2(embeddings)

    if args.num_queries >= count:
        raise SystemExit("num-queries must be smaller than the dataset")

    # Fixed split: query rows held out, gallery = everything else.
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(count)
    query_rows = np.sort(permutation[: args.num_queries])
    gallery_rows = np.sort(permutation[args.num_queries :])
    gallery = np.ascontiguousarray(embeddings[gallery_rows])
    queries = np.ascontiguousarray(embeddings[query_rows])

    out_dir = Path(args.out_dir)
    per_query_dir = out_dir / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)
    np.save(per_query_dir / "query_rows.npy", query_rows)
    np.save(per_query_dir / "gallery_rows.npy", gallery_rows)

    build_threads = args.build_threads if args.build_threads > 0 else faiss.omp_get_max_threads()
    results = []

    def set_threads(n):
        faiss.omp_set_num_threads(n)

    # ---------------- Exact baseline ----------------
    print("building Exact FlatIP...", flush=True)
    set_threads(build_threads)
    start = time.perf_counter()
    exact = faiss.IndexFlatIP(dimension)
    exact.add(gallery)
    build_seconds = time.perf_counter() - start

    set_threads(args.threads)
    exact_indices, search_seconds = timed_search(exact, queries, args.k_gt, args.repeats)
    ground_truth = exact_indices
    np.save(per_query_dir / "exact_top20.npy", ground_truth)

    results.append(
        result_row(
            "Exact FlatIP", {"metric": "cosine"}, build_seconds,
            search_seconds, args.num_queries, exact, ground_truth, exact_indices,
        )
    )
    print(f"exact: {search_seconds / args.num_queries * 1000:.3f} ms/query", flush=True)
    del exact

    # ---------------- HNSW ----------------
    print("building HNSW...", flush=True)
    set_threads(build_threads)
    start = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dimension, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = args.hnsw_ef_construction
    hnsw.add(gallery)
    hnsw_build_seconds = time.perf_counter() - start
    print(f"hnsw build: {hnsw_build_seconds:.1f}s", flush=True)

    set_threads(args.threads)
    for ef_search in hnsw_ef_list:
        hnsw.hnsw.efSearch = ef_search
        indices, search_seconds = timed_search(hnsw, queries, args.k_gt, args.repeats)
        np.save(per_query_dir / f"hnsw_ef{ef_search}_top20.npy", indices)
        results.append(
            result_row(
                "HNSW",
                {"M": args.hnsw_m, "efConstruction": args.hnsw_ef_construction,
                 "efSearch": ef_search},
                hnsw_build_seconds, search_seconds, args.num_queries,
                hnsw, ground_truth, indices,
            )
        )
        print(f"hnsw efSearch={ef_search}: "
              f"recall@10={results[-1]['recall_at_10']:.3f}", flush=True)
    del hnsw

    # ---------------- IVF-PQ ----------------
    print("building IVF-PQ (train + add)...", flush=True)
    set_threads(build_threads)
    sample_rng = np.random.default_rng(args.seed + 1)
    sample_size = min(args.ivf_train_sample, len(gallery_rows))
    train_rows = sample_rng.choice(len(gallery_rows), size=sample_size, replace=False)
    train_vectors = np.ascontiguousarray(gallery[train_rows])

    start = time.perf_counter()
    quantizer = faiss.IndexFlatIP(dimension)
    ivfpq = faiss.IndexIVFPQ(
        quantizer, dimension, args.ivf_nlist, args.pq_m, args.pq_nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    # Unsupervised training on gallery vectors only (no labels involved).
    ivfpq.train(train_vectors)
    ivfpq.add(gallery)
    ivfpq_build_seconds = time.perf_counter() - start
    print(f"ivfpq build: {ivfpq_build_seconds:.1f}s", flush=True)

    set_threads(args.threads)
    for nprobe in nprobe_list:
        ivfpq.nprobe = nprobe
        indices, search_seconds = timed_search(ivfpq, queries, args.k_gt, args.repeats)
        np.save(per_query_dir / f"ivfpq_np{nprobe}_top20.npy", indices)
        results.append(
            result_row(
                "IVF-PQ",
                {"nlist": args.ivf_nlist, "nprobe": nprobe,
                 "m": args.pq_m, "nbits": args.pq_nbits,
                 "train_sample": sample_size},
                ivfpq_build_seconds, search_seconds, args.num_queries,
                ivfpq, ground_truth, indices,
            )
        )
        print(f"ivfpq nprobe={nprobe}: "
              f"recall@10={results[-1]['recall_at_10']:.3f}", flush=True)
    del ivfpq

    # ---------------- LSH ----------------
    for nbits in lsh_nbits_list:
        print(f"building LSH nbits={nbits}...", flush=True)
        set_threads(build_threads)
        start = time.perf_counter()
        # FAISS IndexLSH hashes to binary codes and ranks by Hamming
        # distance, then refines with exact L2. On L2-normalized vectors
        # L2 ranking equals cosine ranking, so results are comparable.
        lsh = faiss.IndexLSH(dimension, nbits)
        lsh.add(gallery)
        lsh_build_seconds = time.perf_counter() - start

        set_threads(args.threads)
        indices, search_seconds = timed_search(lsh, queries, args.k_gt, args.repeats)
        np.save(per_query_dir / f"lsh_nb{nbits}_top20.npy", indices)
        results.append(
            result_row(
                "LSH",
                {"nbits": nbits},
                lsh_build_seconds, search_seconds, args.num_queries,
                lsh, ground_truth, indices,
            )
        )
        print(f"lsh nbits={nbits}: "
              f"recall@10={results[-1]['recall_at_10']:.3f}", flush=True)
        del lsh

    # ---------------- Save aggregate results ----------------
    metadata = {
        "dataset_size": count,
        "gallery_size": int(len(gallery_rows)),
        "num_queries": args.num_queries,
        "dimension": dimension,
        "seed": args.seed,
        "k_gt": args.k_gt,
        "recall_ks": list(RECALL_KS),
        "embedding_file": str(Path(args.emb)),
        "search_threads": args.threads,
        "build_threads": build_threads,
    }
    with (out_dir / "ann_results.json").open("w") as file:
        json.dump({"metadata": metadata, "results": results}, file, indent=2)

    fields = [
        "algorithm", "parameters", "build_seconds", "search_seconds",
        "latency_ms_per_query", "queries_per_second",
        *[f"recall_at_{k}" for k in RECALL_KS],
        "index_size_mb",
    ]
    with (out_dir / "ann_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            csv_row = dict(row)
            csv_row["parameters"] = json.dumps(csv_row["parameters"], sort_keys=True)
            writer.writerow(csv_row)

    print("\nDONE - results in", out_dir, flush=True)


if __name__ == "__main__":
    main()
