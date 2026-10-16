"""Effect of PQ compression on IVF-PQ and HNSW-PQ (every configuration).

Varies only the PQ code size m: the number of sub-quantizers, each storing
an 8-bit code, so every vector takes m bytes. The same m gives both models
the same memory per vector (each sub-quantizer then covers dimension / m
dimensions). The main benchmark uses m = 48 for ViT and 32 for ResNet18;
the default sweep 8,16,32,64,128 divides both 768 and 512.

The search setting is fixed, not swept: nprobe for IVF-PQ and efSearch for
HNSW-PQ are the knee-score picks (scoring.knee_score) of the full-vector
partners IVF-Flat and HNSW in the main benchmark's results for the same
configuration, so each PQ index is compared with its full-vector version at
the same setting. --nprobe / --ef-search override them.

The query/gallery split and the exact top-20 are read from the main
benchmark's results/<config>/per_query/, and the training sample is drawn
exactly as in benchmark_ann.py, so the main benchmark's m reproduces its
numbers.
Run benchmark_ann.py for the configuration first.

Outputs (in results/<config>/pq_sweep/ unless --out-dir is given):
- pq_sweep_results.json / pq_sweep_results.csv
"""

import argparse
import json
import time
from pathlib import Path

# benchmark_ann sets the BLAS thread variables before faiss/numpy load.
from benchmark_ann import CSV_FIELDS, format_metrics, result_row, timed_search, write_csv

import faiss
import numpy as np
import pandas as pd

from datasets import CONFIGS, results_dir
from scoring import knee_score


def knee_pick(key, algorithm, param):
    # Knee-score pick of `algorithm`'s `param` in the main benchmark results.
    path = results_dir(key) / "ann_results.json"
    rows = [{"config": key, "algorithm": r["algorithm"], "param_value": r["parameters"][param],
             **{k: v for k, v in r.items() if k not in ("algorithm", "parameters")}}
            for r in json.loads(path.read_text())["results"] if r["algorithm"] == algorithm]
    sweep = pd.DataFrame(rows)
    return int(sweep.loc[knee_score(sweep).idxmax(), "param_value"])


def run_sweep(key, cfg, args):
    per_query_dir = results_dir(key) / "per_query"
    split_files = [per_query_dir / f"{name}.npy" for name in ("query_rows", "gallery_rows", "exact_top20")]
    missing = [path.name for path in split_files if not path.exists()]
    if missing or not (results_dir(key) / "ann_results.json").exists():
        raise SystemExit(f"[{key}] main benchmark results missing in {results_dir(key)}: "
                         f"run benchmark_ann.py --config {key} first")
    query_rows, gallery_rows, ground_truth = (np.load(path) for path in split_files)

    nprobe = args.nprobe or knee_pick(key, "IVF-Flat", "nprobe")
    ef_search = args.ef_search or knee_pick(key, "HNSW", "efSearch")

    embeddings = np.ascontiguousarray(np.load(cfg["emb"]))
    count, dimension = embeddings.shape
    norms = np.linalg.norm(embeddings[:1024], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("note: vectors not normalized - normalizing in memory")
        faiss.normalize_L2(embeddings)
    gallery = np.ascontiguousarray(embeddings[gallery_rows])
    queries = np.ascontiguousarray(embeddings[query_rows])
    num_queries = len(queries)

    # Same training sample as benchmark_ann.py.
    sample_rng = np.random.default_rng(args.seed + 1)
    sample_size = min(args.ivf_train_sample, len(gallery_rows))
    train_rows = sample_rng.choice(len(gallery_rows), size=sample_size, replace=False)
    train_vectors = np.ascontiguousarray(gallery[train_rows])

    m_list = [int(x) for x in args.m.split(",")]
    build_threads = args.build_threads if args.build_threads > 0 else faiss.omp_get_max_threads()
    results = []

    print(f"\n===== {cfg['label']}: PQ sweep, {len(gallery):,} x {dimension}-d, "
          f"nprobe={nprobe}, efSearch={ef_search} =====", flush=True)
    for pq_m in m_list:
        if dimension % pq_m != 0:
            print(f"skipping m={pq_m}: does not divide the dimension ({dimension})", flush=True)
            continue
        dsub = dimension // pq_m   # dimensions per sub-quantizer

        # ---------------- IVF-PQ ----------------
        print(f"building IVF-PQ m={pq_m} (dsub={dsub})...", flush=True)
        faiss.omp_set_num_threads(build_threads)
        start = time.perf_counter()
        quantizer = faiss.IndexFlatIP(dimension)
        ivfpq = faiss.IndexIVFPQ(
            quantizer, dimension, args.ivf_nlist, pq_m, args.pq_nbits,
            faiss.METRIC_INNER_PRODUCT,
        )
        ivfpq.train(train_vectors)
        ivfpq.add(gallery)
        build_seconds = time.perf_counter() - start

        faiss.omp_set_num_threads(args.threads)
        ivfpq.nprobe = nprobe
        indices, search_seconds = timed_search(ivfpq, queries, args.k_gt, args.repeats)
        results.append(result_row(
            "IVF-PQ",
            {"nlist": args.ivf_nlist, "nprobe": nprobe, "m": pq_m, "dsub": dsub,
             "nbits": args.pq_nbits, "train_sample": sample_size},
            build_seconds, search_seconds, num_queries, ivfpq, ground_truth, indices,
        ))
        print(f"ivfpq m={pq_m}: {format_metrics(results[-1])}", flush=True)
        del ivfpq

        # ---------------- HNSW-PQ ----------------
        print(f"building HNSW-PQ m={pq_m} (dsub={dsub})...", flush=True)
        faiss.omp_set_num_threads(build_threads)
        start = time.perf_counter()
        # L2 metric, as in benchmark_ann.py (IndexHNSWPQ with inner product
        # returns wrong neighbours in FAISS 1.15; L2 = cosine ranking here).
        hnswpq = faiss.IndexHNSWPQ(dimension, pq_m, args.hnsw_m, args.pq_nbits)
        hnswpq.hnsw.efConstruction = args.hnsw_ef_construction
        hnswpq.train(train_vectors)
        hnswpq.add(gallery)
        build_seconds = time.perf_counter() - start

        faiss.omp_set_num_threads(args.threads)
        hnswpq.hnsw.efSearch = ef_search
        indices, search_seconds = timed_search(hnswpq, queries, args.k_gt, args.repeats)
        results.append(result_row(
            "HNSW-PQ",
            {"M": args.hnsw_m, "efConstruction": args.hnsw_ef_construction,
             "efSearch": ef_search, "m": pq_m, "dsub": dsub,
             "nbits": args.pq_nbits, "train_sample": sample_size},
            build_seconds, search_seconds, num_queries, hnswpq, ground_truth, indices,
        ))
        print(f"hnswpq m={pq_m}: {format_metrics(results[-1])}", flush=True)
        del hnswpq

    out_dir = Path(args.out_dir) / key if args.out_dir else results_dir(key) / "pq_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": key,
        "label": cfg["label"],
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "gallery_size": int(len(gallery_rows)),
        "num_queries": num_queries,
        "dimension": dimension,
        "seed": args.seed,
        "m": m_list,
        "nprobe": nprobe,
        "efSearch": ef_search,
        "search_threads": args.threads,
        "build_threads": build_threads,
    }
    with (out_dir / "pq_sweep_results.json").open("w") as file:
        json.dump({"metadata": metadata, "results": results}, file, indent=2)
    write_csv(out_dir / "pq_sweep_results.csv", results, CSV_FIELDS)
    print(f"results in {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--m", default="8,16,32,64,128",
                        help="PQ sub-quantizers (bytes per vector); must divide the dimension")
    parser.add_argument("--nprobe", type=int, help="override the IVF-Flat knee pick")
    parser.add_argument("--ef-search", type=int, help="override the HNSW knee pick")
    parser.add_argument("--out-dir", help="write to <out-dir>/<config>/ instead of results/<config>/pq_sweep/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-gt", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=0)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--ivf-nlist", type=int, default=256)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--ivf-train-sample", type=int, default=50000)
    args = parser.parse_args()

    keys = list(CONFIGS) if args.config == "all" else [args.config]
    for key in keys:
        run_sweep(key, CONFIGS[key], args)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
