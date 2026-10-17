"""ANN index robustness under query corruption (IVF / HNSW, with and without PQ).

Extends the exact-search corruption experiment (pipeline/corruption.py, which
stays untouched) to the four ANN indexes of the main benchmark: IVF-Flat,
IVF-PQ, HNSW and HNSW-PQ. The corrupted query vectors and their exact
references are not recomputed: this script consumes the cache that
corruption.py already wrote to results/<config>/robustness/, so model
robustness and index robustness are measured on exactly the same queries,
corruptions and ground truth.

Each index is rebuilt with the main benchmark's construction parameters
(recovered from results/<config>/ann_results.json) and searched at the
knee-score setting (scoring.knee_score) of its full-vector partner, following
the pq_sweep.py convention: IVF-Flat and IVF-PQ share nprobe, HNSW and
HNSW-PQ share efSearch, so every PQ index is compared with its uncompressed
version at the same search effort. LSH is out of scope (not part of the
IVF/HNSW four-way comparison) and Exact FlatIP is the reference of every
metric family, not a competitor.

Three metric families per query variant and K = 5/10/20:
- fidelity_*    corrupted ANN vs exact search of the same corrupted query
                -> pure ANN approximation error under corruption
- end_to_end_*  corrupted ANN vs exact search of the clean query
                -> total retrieval degradation (model + index)
- stability_*   corrupted ANN vs the same index on the clean query
                -> how much the index's own output changes

The `original` level (the uncorrupted query, present in the cache) anchors
the sanity checks: stability must be exactly 1.0 (the index is compared with
itself) and fidelity must equal end_to_end (both exact references agree).

Outputs per configuration in results/<config>/index_robustness/:
- index_robustness_summary.csv          mean/std per algorithm x transform x level
- index_robustness_params.json          settings, sources and self-checks
- per_query/index_robustness_per_variant.csv   one row per algorithm x variant
- per_query/<method>_<setting>_top<K>.npy      ANN neighbours per variant

And results/index_robustness_combined.csv across all configurations.

Example (from the repository root, after pipeline/corruption.py):
    python experiments/ann-comparison/pipeline/index_robustness.py --config all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Same BLAS pinning as benchmark_ann.py: must be set before faiss/numpy load.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

PIPELINE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PIPELINE_DIR.parent
# datasets.py / benchmark_ann.py / scoring.py live in the experiment folder.
for _path in (str(EXPERIMENT_DIR), str(PIPELINE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import faiss
import numpy as np
import pandas as pd

from benchmark_ann import map_at_k, precision_at_k, recall_at_k, timed_search
from datasets import CONFIGS, RESULTS_DIR, results_dir
from scoring import knee_score

METHOD_NAMES = {
    "ivfflat": "IVF-Flat",
    "ivfpq": "IVF-PQ",
    "hnsw": "HNSW",
    "hnswpq": "HNSW-PQ",
}
METRIC_FAMILIES = ("fidelity", "end_to_end", "stability")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--methods", default="ivfflat,ivfpq,hnsw,hnswpq",
                        help="comma-separated subset of: " + ",".join(METHOD_NAMES))
    parser.add_argument("--nprobe", type=int,
                        help="override the IVF-Flat knee pick (IVF-Flat and IVF-PQ)")
    parser.add_argument("--ef-search", type=int,
                        help="override the HNSW knee pick (HNSW and HNSW-PQ)")
    parser.add_argument("--k-gt", type=int, default=20,
                        help="must match the depth cached by pipeline/corruption.py")
    parser.add_argument("--metric-ks", default="5,10,20")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=0)
    parser.add_argument("--no-save-topk", action="store_true",
                        help="do not save per-method top-k arrays to per_query/")
    parser.add_argument("--out-dir",
                        help="write to <out-dir>/<config>/ instead of results/<config>/index_robustness/")
    return parser.parse_args()


def select_methods(requested):
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in METHOD_NAMES]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown} - choose from {sorted(METHOD_NAMES)}")
    return [METHOD_NAMES[name] for name in names]


def knee_row(key, algorithm, rows):
    """The sweep row of `algorithm` picked by scoring.knee_score."""
    frame = pd.DataFrame([
        {"config": key, "algorithm": row["algorithm"], **row}
        for row in rows if row["algorithm"] == algorithm
    ])
    if frame.empty:
        raise SystemExit(f"[{key}] ann_results.json has no {algorithm} rows")
    return frame.loc[knee_score(frame).idxmax()]


def matched_row(key, algorithm, rows, **parameters):
    """The benchmark row of `algorithm` at `parameters`; any row of the sweep if
    none matches (construction parameters are constant across a sweep, so the
    first row is a safe fallback when an override picks an unswept value)."""
    candidates = [row for row in rows if row["algorithm"] == algorithm]
    if not candidates:
        raise SystemExit(f"[{key}] ann_results.json has no {algorithm} rows")
    for row in candidates:
        if all(row["parameters"].get(name) == value for name, value in parameters.items()):
            return row
    return candidates[0]


def select_settings(key, rows, args):
    """One complete parameter set per method, recovered from the benchmark.

    Search settings follow the pq_sweep.py convention: the knee-score pick of
    the full-vector partner (IVF-Flat for IVF-PQ, HNSW for HNSW-PQ), so each
    PQ index runs at the same search effort as its uncompressed version.
    """
    ivfflat = knee_row(key, "IVF-Flat", rows)
    hnsw = knee_row(key, "HNSW", rows)
    nprobe = args.nprobe or int(ivfflat["parameters"]["nprobe"])
    ef_search = args.ef_search or int(hnsw["parameters"]["efSearch"])
    ivfpq = matched_row(key, "IVF-PQ", rows, nprobe=nprobe)
    hnswpq = matched_row(key, "HNSW-PQ", rows, efSearch=ef_search)

    settings = {
        "IVF-Flat": {
            "nlist": int(ivfflat["parameters"]["nlist"]),
            "nprobe": nprobe,
            "train_sample": int(ivfflat["parameters"]["train_sample"]),
        },
        "IVF-PQ": {
            "nlist": int(ivfpq["parameters"]["nlist"]),
            "nprobe": nprobe,
            "m": int(ivfpq["parameters"]["m"]),
            "nbits": int(ivfpq["parameters"]["nbits"]),
            "train_sample": int(ivfpq["parameters"]["train_sample"]),
        },
        "HNSW": {
            "M": int(hnsw["parameters"]["M"]),
            "efConstruction": int(hnsw["parameters"]["efConstruction"]),
            "efSearch": ef_search,
        },
        "HNSW-PQ": {
            "M": int(hnswpq["parameters"]["M"]),
            "efConstruction": int(hnswpq["parameters"]["efConstruction"]),
            "efSearch": ef_search,
            "m": int(hnswpq["parameters"]["m"]),
            "nbits": int(hnswpq["parameters"]["nbits"]),
            "train_sample": int(hnswpq["parameters"]["train_sample"]),
        },
    }
    index_sizes = {
        "IVF-Flat": ivfflat.get("index_size_mb"),
        "IVF-PQ": ivfpq.get("index_size_mb"),
        "HNSW": hnsw.get("index_size_mb"),
        "HNSW-PQ": hnswpq.get("index_size_mb"),
    }
    # All trained methods must share the benchmark's training sample.
    samples = {settings[name]["train_sample"]
               for name in ("IVF-Flat", "IVF-PQ", "HNSW-PQ")}
    if len(samples) != 1:
        raise SystemExit(f"[{key}] benchmark rows disagree on train_sample: {sorted(samples)}")
    return settings, index_sizes


def load_benchmark(key):
    """Benchmark metadata, result rows and the fixed query/gallery split."""
    path = results_dir(key) / "ann_results.json"
    if not path.exists():
        raise SystemExit(f"[{key}] {path} missing - run benchmark_ann.py --config {key} first")
    data = json.loads(path.read_text())
    metadata, rows = data["metadata"], data["results"]

    per_query_dir = results_dir(key) / "per_query"
    query_path = per_query_dir / "query_rows.npy"
    gallery_path = per_query_dir / "gallery_rows.npy"
    if query_path.exists() and gallery_path.exists():
        query_rows = np.load(query_path)
        gallery_rows = np.load(gallery_path)
        split_source = "per_query/query_rows.npy + gallery_rows.npy"
    else:
        permutation = np.random.default_rng(metadata["seed"]).permutation(metadata["dataset_size"])
        query_rows = np.sort(permutation[: metadata["num_queries"]])
        gallery_rows = np.sort(permutation[metadata["num_queries"] :])
        split_source = f"reproduced from ann_results.json (seed {metadata['seed']})"
    if len(query_rows) != metadata["num_queries"] or len(gallery_rows) != metadata["gallery_size"]:
        raise SystemExit(f"[{key}] recovered split does not match the benchmark metadata")
    return metadata, rows, query_rows, gallery_rows, split_source


def load_corruption_cache(key, dimension):
    """The vectors, variant table and exact references written by corruption.py."""
    rob_dir = results_dir(key) / "robustness"
    required = [
        "corrupted_query_embeddings.npy", "corrupted_query_variants.csv",
        "corrupted_query_top20.npy", "corrupted_query_clean_top20.npy",
        "corruption_params.json",
    ]
    missing = [name for name in required if not (rob_dir / name).exists()]
    if missing:
        raise SystemExit(f"[{key}] corruption cache missing in {rob_dir}: {missing}\n"
                         f"run pipeline/corruption.py --config {key} first")
    params = json.loads((rob_dir / "corruption_params.json").read_text())
    vectors = np.load(rob_dir / "corrupted_query_embeddings.npy")
    # keep_default_na=False: the original level is the empty string and must
    # survive grouping.
    variants = pd.read_csv(rob_dir / "corrupted_query_variants.csv", keep_default_na=False)
    variant_topk = np.load(rob_dir / "corrupted_query_top20.npy")
    clean_topk = np.load(rob_dir / "corrupted_query_clean_top20.npy")

    num_queries = int(params["num_queries"])
    if int(params["dimension"]) != dimension or vectors.shape[1] != dimension:
        raise SystemExit(f"[{key}] corruption cache dimension does not match the registry")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise SystemExit(f"[{key}] corrupted vectors are not finite float32")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise SystemExit(f"[{key}] corrupted vectors are not L2-normalized")
    if len(variants) != len(vectors) or len(variant_topk) != len(vectors):
        raise SystemExit(f"[{key}] corruption cache rows are misaligned")
    if len(clean_topk) != num_queries:
        raise SystemExit(f"[{key}] clean exact reference does not cover every query")
    if variant_topk.shape[1] != clean_topk.shape[1] or int(params["k_gt"]) != variant_topk.shape[1]:
        raise SystemExit(f"[{key}] exact reference depth is inconsistent")
    if not variants["query"].between(0, num_queries - 1).all():
        raise SystemExit(f"[{key}] variant table has out-of-range query ids")
    originals = variants[variants["transform"] == "original"]
    if len(originals) != num_queries or not np.array_equal(
            np.sort(originals["query"].to_numpy()), np.arange(num_queries)):
        raise SystemExit(f"[{key}] variant table must hold exactly one original per query")

    variants = variants.copy()
    # Level order within each transform, as defined by pipeline/perturbations.py.
    variants["variant_order"] = variants.groupby(["query", "transform"], sort=False).cumcount()
    return params, vectors, variants, variant_topk, clean_topk


def build_index(algorithm, dimension, gallery, train_vectors, parameters):
    """Rebuild one benchmark index with its recorded construction parameters."""
    if algorithm == "IVF-Flat":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, parameters["nlist"], faiss.METRIC_INNER_PRODUCT,
        )
        index.train(train_vectors)
        index.add(gallery)
        index.nprobe = parameters["nprobe"]
    elif algorithm == "IVF-PQ":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(
            quantizer, dimension, parameters["nlist"], parameters["m"],
            parameters["nbits"], faiss.METRIC_INNER_PRODUCT,
        )
        index.train(train_vectors)
        index.add(gallery)
        index.nprobe = parameters["nprobe"]
    elif algorithm == "HNSW":
        index = faiss.IndexHNSWFlat(dimension, parameters["M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = parameters["efConstruction"]
        index.add(gallery)
        index.hnsw.efSearch = parameters["efSearch"]
    elif algorithm == "HNSW-PQ":
        # L2, as in benchmark_ann.py: IndexHNSWPQ with METRIC_INNER_PRODUCT
        # returns wrong neighbours in FAISS 1.15; L2 ranking equals cosine
        # ranking on the unit vectors used here.
        index = faiss.IndexHNSWPQ(
            dimension, parameters["m"], parameters["M"], parameters["nbits"],
        )
        index.hnsw.efConstruction = parameters["efConstruction"]
        index.train(train_vectors)
        index.add(gallery)
        index.hnsw.efSearch = parameters["efSearch"]
    else:
        raise ValueError(algorithm)
    return index


def setting_label(algorithm, parameters):
    if algorithm in ("IVF-Flat", "IVF-PQ"):
        return f"nprobe={parameters['nprobe']}"
    return f"efSearch={parameters['efSearch']}"


def topk_stem(algorithm, parameters):
    if algorithm == "IVF-Flat":
        return f"ivfflat_np{parameters['nprobe']}"
    if algorithm == "IVF-PQ":
        return f"ivfpq_np{parameters['nprobe']}"
    if algorithm == "HNSW":
        return f"hnsw_ef{parameters['efSearch']}"
    return f"hnswpq_ef{parameters['efSearch']}"


def run_config(key, cfg, args, methods):
    metric_ks = args.metric_ks
    metadata, rows, query_rows, gallery_rows, split_source = load_benchmark(key)
    settings, index_sizes = select_settings(key, rows, args)

    embeddings = np.ascontiguousarray(np.load(cfg["emb"]))
    ids = np.load(cfg["ids"])
    count, dimension = embeddings.shape
    if dimension != cfg["dimension"] or int(metadata["dimension"]) != dimension:
        raise SystemExit(f"[{key}] embedding dimension does not match the registry/benchmark")
    if count != int(metadata["dataset_size"]) or len(ids) != count:
        raise SystemExit(f"[{key}] embeddings/ids do not match the benchmark dataset size")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
        raise SystemExit(
            f"[{key}] embeddings are not L2-normalized; the corruption cache was "
            "built against the on-disk gallery, so it must not be changed here"
        )

    params, vectors, variants, variant_topk, clean_topk = load_corruption_cache(key, dimension)
    k_gt = variant_topk.shape[1]
    if args.k_gt != k_gt:
        raise SystemExit(f"[{key}] --k-gt {args.k_gt} but the corruption cache holds k={k_gt}")
    if max(metric_ks) > k_gt:
        raise SystemExit(f"[{key}] metric Ks exceed the cached exact depth {k_gt}")

    corruption_rows = np.asarray(params["query_rows"], dtype=np.int64)
    if not set(corruption_rows.tolist()) <= set(query_rows.tolist()):
        raise SystemExit(f"[{key}] corruption queries are not a subset of the benchmark queries")

    gallery = np.ascontiguousarray(embeddings[gallery_rows])
    clean = np.ascontiguousarray(embeddings[corruption_rows])
    del embeddings

    # Self-check: the recovered gallery must reproduce the cached clean exact
    # reference (same split, same ordering -> identical neighbours).
    exact = faiss.IndexFlatIP(dimension)
    exact.add(gallery)
    _, recomputed = exact.search(clean, k_gt)
    agreement = recall_at_k(clean_topk, recomputed, k_gt)
    del exact
    if agreement < 0.99:
        raise SystemExit(f"[{key}] recovered gallery reproduces the cached clean exact "
                         f"reference only at recall@{k_gt}={agreement:.4f}")

    num_queries = len(corruption_rows)
    num_variants = len(vectors)
    query_of_variant = variants["query"].to_numpy(dtype=np.int64)
    cosines = np.einsum("ij,ij->i", clean[query_of_variant], vectors)
    originals_mask = (variants["transform"] == "original").to_numpy()
    original_positions = np.full(num_queries, -1, dtype=np.int64)
    original_positions[query_of_variant[originals_mask]] = np.flatnonzero(originals_mask)

    train_vectors = None
    if any(name in ("IVF-Flat", "IVF-PQ", "HNSW-PQ") for name in methods):
        # Same training sample as benchmark_ann.py / pq_sweep.py.
        sample_size = settings["IVF-Flat"]["train_sample"]
        if sample_size > len(gallery_rows):
            raise SystemExit(f"[{key}] train_sample {sample_size} exceeds the gallery")
        sample_rng = np.random.default_rng(int(metadata["seed"]) + 1)
        train_rows = sample_rng.choice(len(gallery_rows), size=sample_size, replace=False)
        train_vectors = np.ascontiguousarray(gallery[train_rows])

    build_threads = args.build_threads if args.build_threads > 0 else faiss.omp_get_max_threads()
    out_dir = Path(args.out_dir) / key if args.out_dir else results_dir(key) / "index_robustness"
    per_query_dir = out_dir / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{key}] {num_queries} queries x {num_variants // num_queries} variants, "
          f"{len(gallery_rows):,} gallery, {dimension}-d, exact reference "
          f"reproduced at recall@{k_gt}={agreement:.6f}", flush=True)

    variant_meta = variants[["query", "image_id", "transform", "level", "label", "variant_order"]]
    frames, timings = [], {}
    for algorithm in methods:
        parameters = settings[algorithm]
        label = setting_label(algorithm, parameters)
        print(f"[{key}] building {algorithm} ({label})...", flush=True)
        faiss.omp_set_num_threads(build_threads)
        start = time.perf_counter()
        index = build_index(algorithm, dimension, gallery, train_vectors, parameters)
        build_seconds = time.perf_counter() - start
        print(f"[{key}] {algorithm} build: {build_seconds:.1f}s", flush=True)

        faiss.omp_set_num_threads(args.threads)
        indices, search_seconds = timed_search(index, vectors, k_gt, args.repeats)
        unfilled = int((indices < 0).sum())
        if unfilled:
            print(f"[{key}] {algorithm}: {unfilled} of {indices.size} neighbour slots unfilled",
                  flush=True)
        if not args.no_save_topk:
            np.save(per_query_dir / f"{topk_stem(algorithm, parameters)}_top{k_gt}.npy", indices)
        del index

        # The three references, all aligned with the variant rows:
        # fidelity: exact neighbours of the same corrupted query;
        # end_to_end: exact neighbours of the clean query;
        # stability: this index's own neighbours of the clean query.
        stability_gt = indices[original_positions][query_of_variant]
        references = (
            ("fidelity", variant_topk),
            ("end_to_end", clean_topk[query_of_variant]),
            ("stability", stability_gt),
        )
        out_rows = []
        for position in range(num_variants):
            actual = indices[position:position + 1]
            meta = variant_meta.iloc[position]
            row = {
                "config": key,
                "model": cfg["model"],
                "dataset": cfg["dataset"],
                "algorithm": algorithm,
                "setting": label,
                "query": int(meta["query"]),
                "image_id": int(meta["image_id"]),
                "transform": meta["transform"],
                "level": meta["level"],
                "label": meta["label"],
                "variant_order": int(meta["variant_order"]),
                "cosine": round(float(cosines[position]), 6),
            }
            for prefix, reference in references:
                for k in metric_ks:
                    gt = reference[position:position + 1]
                    row[f"{prefix}_recall_at_{k}"] = round(recall_at_k(gt, actual, k), 6)
                    row[f"{prefix}_precision_at_{k}"] = round(precision_at_k(gt, actual, k), 6)
                    row[f"{prefix}_map_at_{k}"] = round(map_at_k(gt, actual, k), 6)
            out_rows.append(row)
        frame = pd.DataFrame(out_rows)
        frames.append(frame)
        timings[algorithm] = {
            "build_seconds": round(build_seconds, 3),
            "search_seconds": round(search_seconds, 3),
            "latency_ms_per_variant": round(search_seconds * 1000 / num_variants, 4),
            "variants_per_second": round(num_variants / search_seconds, 1),
        }

        # Original-level self-checks.
        original_rows = frame[originals_mask]
        for k in metric_ks:
            if original_rows[f"stability_recall_at_{k}"].min() != 1.0:
                raise SystemExit(f"[{key}] {algorithm}: original stability recall@{k} below 1.0")
        print(f"[{key}] {algorithm} original self-check: stability 1.0, "
              f"fidelity recall@{max(metric_ks)} min "
              f"{original_rows[f'fidelity_recall_at_{max(metric_ks)}'].min():.4f}, "
              f"end_to_end recall@{max(metric_ks)} min "
              f"{original_rows[f'end_to_end_recall_at_{max(metric_ks)}'].min():.4f}", flush=True)
        print(f"[{key}] {algorithm} {label}: search {search_seconds:.1f}s "
              f"({timings[algorithm]['latency_ms_per_variant']:.3f} ms/variant)", flush=True)

    pd.concat(frames, ignore_index=True).to_csv(
        per_query_dir / "index_robustness_per_variant.csv", index=False
    )

    metric_cols = ["cosine"] + [
        f"{prefix}_{metric}_at_{k}"
        for prefix in METRIC_FAMILIES
        for metric in ("recall", "precision", "map")
        for k in metric_ks
    ]
    summaries = []
    for algorithm, frame in zip(methods, frames):
        summary = (
            frame.groupby(["model", "dataset", "transform", "level", "label"], sort=False)[metric_cols]
            .agg(["mean", "std"])
            .round(6)
        )
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary = summary.reset_index()
        summary.insert(0, "config", key)
        summary["algorithm"] = algorithm
        summary["setting"] = setting_label(algorithm, settings[algorithm])
        summary["parameters"] = json.dumps(settings[algorithm], sort_keys=True)
        summary["num_queries"] = num_queries
        summary["build_seconds"] = timings[algorithm]["build_seconds"]
        summary["search_seconds"] = timings[algorithm]["search_seconds"]
        summary["latency_ms_per_variant"] = timings[algorithm]["latency_ms_per_variant"]
        summary["variants_per_second"] = timings[algorithm]["variants_per_second"]
        summary["index_size_mb"] = index_sizes.get(algorithm)
        summaries.append(summary)
    summary_frame = pd.concat(summaries, ignore_index=True)
    leading = [
        "config", "model", "dataset", "algorithm", "setting", "parameters",
        "num_queries", "build_seconds", "search_seconds", "latency_ms_per_variant",
        "variants_per_second", "index_size_mb", "transform", "level", "label",
    ]
    summary_frame = summary_frame[leading + [c for c in summary_frame.columns if c not in leading]]
    summary_frame.to_csv(out_dir / "index_robustness_summary.csv", index=False)

    out_params = {
        "config": key,
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "dimension": dimension,
        "num_queries": num_queries,
        "num_variants": num_variants,
        "variants_per_query": num_variants // num_queries,
        "k_gt": k_gt,
        "metric_ks": list(metric_ks),
        "benchmark_seed": metadata["seed"],
        "gallery_size": int(len(gallery_rows)),
        "split_source": split_source,
        "embedding_file": str(cfg["emb"]),
        "ann_results_file": str(results_dir(key) / "ann_results.json"),
        "corruption_cache_dir": str(results_dir(key) / "robustness"),
        "setting_source": (
            "knee-score picks (scoring.knee_score) of the full-vector partners in "
            "ann_results.json; IVF-PQ shares IVF-Flat's nprobe and HNSW-PQ shares "
            "HNSW's efSearch (pq_sweep.py convention)"
        ),
        "nprobe_override": args.nprobe,
        "ef_search_override": args.ef_search,
        "settings": settings,
        "methods": methods,
        "search_threads": args.threads,
        "build_threads": build_threads,
        "repeats": args.repeats,
        "faiss_version": faiss.__version__,
        "self_check": {
            "clean_exact_reference_recall": round(float(agreement), 6),
            "original_cosine_min": round(float(cosines[originals_mask].min()), 6),
        },
        "topk_files": {} if args.no_save_topk else {
            algorithm: f"per_query/{topk_stem(algorithm, settings[algorithm])}_top{k_gt}.npy"
            for algorithm in methods
        },
    }
    (out_dir / "index_robustness_params.json").write_text(json.dumps(out_params, indent=2))
    print(f"[{key}] results in {out_dir}", flush=True)
    return summary_frame


def write_combined():
    # Rebuild the cross-configuration table from whatever results exist, so a
    # single-configuration run still refreshes it (benchmark_ann.py behaviour).
    combined = []
    for key in CONFIGS:
        path = results_dir(key) / "index_robustness" / "index_robustness_summary.csv"
        if not path.exists():
            continue
        combined.append(pd.read_csv(path, keep_default_na=False))
    if not combined:
        return
    out_path = RESULTS_DIR / "index_robustness_combined.csv"
    pd.concat(combined, ignore_index=True).to_csv(out_path, index=False)
    print(f"combined table: {out_path}", flush=True)


def main():
    args = parse_args()
    args.metric_ks = [int(x) for x in args.metric_ks.split(",")]
    methods = select_methods(args.methods)

    keys = list(CONFIGS) if args.config == "all" else [args.config]
    for key in keys:
        if not (results_dir(key) / "robustness" / "corruption_params.json").exists():
            print(f"[{key}] no corruption cache - skipping "
                  f"(run pipeline/corruption.py --config {key})", flush=True)
            continue
        run_config(key, CONFIGS[key], args, methods)
    write_combined()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
