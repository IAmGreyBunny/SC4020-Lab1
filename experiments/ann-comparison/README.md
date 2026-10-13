# ann-comparison

One folder for the whole ANN pipeline on both team datasets, so the
algorithms can be compared side by side:

| Key | Dataset | Embedding model | Dim | Vectors | Gallery |
|---|---|---|---:|---:|---:|
| `coco` | COCO unlabeled2017 | ViT-B/16 | 768 | 123,403 | 122,403 |
| `zappos` | UT-Zap50K | ResNet18 | 512 | 50,025 | 49,025 |

Methods: Exact FlatIP (ground truth) vs HNSW vs IVF-PQ vs LSH, with
Recall@K, Precision@K and mAP@K (K = 5/10/20), latency, QPS, build time
and index size.

This folder merges `experiments/vit-coco-faiss/` and
`experiments/resnet18-zappos/`. Those folders are left unchanged; the
pipeline scripts here are copies with only their output paths repointed.
The robustness experiment (`experiments/robustness-comparison/`) is
separate and not included.

## Pipeline

```text
COCO unlabeled2017 JPGs              UT-Zap50K JPGs
  -> ViT-B/16 (head removed)           -> ResNet18 (fc removed)
  -> 768-d L2-normalized               -> 512-d L2-normalized
  -> data/circo/vit_*.npy              -> data/zappos/resnet_*.npy
        \                                  /
         benchmark_ann.py --dataset all          (same protocol, one script)
                         |
         results/<dataset>/ann_results.*  +  results/combined_ann_results.csv
                         |
         compare_methods.ipynb                    (both datasets in one notebook)
```

## Files

```text
experiments/ann-comparison/
├── README.md
├── requirements.txt
├── datasets.py               # dataset registry: paths, model, dim, pq-m
├── benchmark_ann.py          # ONE benchmark for every dataset
├── compare_methods.ipynb     # ONE notebook comparing every dataset
├── pipeline/
│   ├── coco/                 # from vit-coco-faiss
│   │   ├── unzip_coco.py           # extract unlabeled2017.zip
│   │   ├── extract_embeddings.py   # ViT-B/16 embeddings (resumable)
│   │   ├── validate_embeddings.py  # shape/dtype/norm/id checks
│   │   └── exact_search.py         # standalone exact-search baseline
│   └── zappos/               # from resnet18-zappos
│       ├── common.py               # path discovery + deterministic ordering
│       ├── extract_features.py     # ResNet18 embeddings
│       ├── search.py               # single-query retrieval check
│       └── evaluate.py             # label-based evaluation
└── results/
    ├── combined_ann_results.csv    # every row, with a `dataset` column
    ├── coco/                       # ann_results.json/.csv, visual-comparisons/
    └── zappos/                     # ann_results.json/.csv, visual-comparisons/
        (per_query/ is generated locally and not committed)
```

## Setup

From the repository root:

```bash
python -m pip install -r experiments/ann-comparison/requirements.txt
```

For NVIDIA GPU embedding extraction, install the CUDA build of PyTorch
for your machine instead of the default wheel.

## Step 1 - Obtain the embeddings

The benchmark only needs the two `.npy` files per dataset. All `.npy`
files under `data/` are ignored by Git and are shared separately.

### COCO (ViT) -> `data/circo/`

**Option A - download** `vit_embeddings.npy`, `vit_image_ids.npy` and
`vit_metadata.json` (~363 MB) from the team's shared Drive folder (link in
`experiments/vit-coco-faiss/README.md`) into `data/circo/`.

**Option B - generate** (about 24 min on a GPU):

```bash
python experiments/ann-comparison/pipeline/coco/unzip_coco.py \
  --zip /path/to/unlabeled2017.zip --dest data/circo --workers 6

python experiments/ann-comparison/pipeline/coco/extract_embeddings.py \
  --images-dir data/circo --output-dir data/circo --prefix vit --batch-size 8
```

Then validate:

```bash
python experiments/ann-comparison/pipeline/coco/validate_embeddings.py \
  --emb data/circo/vit_embeddings.npy --ids data/circo/vit_image_ids.npy \
  --meta data/circo/vit_metadata.json --expect-count 123403 --images-dir data/circo
```

### Zappos (ResNet18) -> `data/zappos/`

**Option A - use the handoff files** `resnet_embeddings.npy` and
`resnet_image_ids.npy` (~98 MB) placed into `data/zappos/`.

**Option B - generate**:

```bash
python experiments/ann-comparison/pipeline/zappos/extract_features.py \
  --images-dir data/zappos/ut-zap50k-images/ut-zap50k-images \
  --output-dir data/zappos --prefix resnet
```

`common.py` still defaults to the original author's local dataset path, so
pass `--images-dir` explicitly.

## Step 2 - Obtain the images (notebook grids only)

The benchmark itself needs only the embeddings. The notebook's
same-query image grids also need:

- COCO: the JPGs in `data/circo/` (or `data/circo/unlabeled2017/`)
- Zappos: `data/zappos/ut-zap50k-images/ut-zap50k-images/` - download
  from Kaggle (`aryashah2k/large-shoe-dataset-ut-zappos50k`) as described
  in `experiments/resnet18-zappos/README.md`

## Step 3 - Run the benchmark

```bash
# both datasets (a dataset whose embeddings are missing is skipped)
python experiments/ann-comparison/benchmark_ann.py

# or one at a time
python experiments/ann-comparison/benchmark_ann.py --dataset coco
python experiments/ann-comparison/benchmark_ann.py --dataset zappos
```

Every run rebuilds `results/combined_ann_results.csv` from all available
`results/*/ann_results.json`, so running one dataset still refreshes the
combined table.

The protocol is identical for both datasets: 1,000 held-out queries
(seed 42), exact top-20 ground truth, single-threaded search, all-core
builds, the same HNSW / IVF / LSH sweeps. The only differences are forced
by the data and live in `datasets.py`:

- `pq-m` 48 for COCO (768-d) and 32 for Zappos (512-d), because FAISS
  needs `dimension % pq_m == 0`.
- IVF-PQ trains on a 50k gallery sample for COCO, and on the full gallery
  (49,025 rows) for Zappos.

Searches are timed single-threaded. The script also pins the BLAS thread
pool (`OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` = 1), because exact
search runs through BLAS, which ignores FAISS's own thread setting. Without
the pin, exact search silently used every core on Windows and looked
faster than HNSW.

### Metrics

All accuracy metrics treat the **exact top-K** as the relevant set for
each query, at K = 5, 10 and 20:

- **Recall@K**: the fraction of the exact top-K that the method also
  returns.
- **Precision@K**: the fraction of the method's top-K that is in the
  exact top-K. Both lists hold K items, so it is numerically equal to
  Recall@K here. It is reported separately for completeness.
- **mAP@K**: the mean over queries of
  AP@K = (1/K) · Σ<sub>i≤K</sub> Precision@i · rel(i).
  It is rank-aware: true neighbours returned lower down the list score
  less.

Exact search scores 1.0 on all three by definition. Padding (`-1`, no
result returned) never counts as relevant.

Outputs per dataset in `results/<dataset>/`:

| File | Contents |
|---|---|
| `ann_results.json` / `.csv` | per method/setting: Recall/Precision/mAP @5/10/20, latency, QPS, build time, index size |
| `per_query/exact_top20.npy` | exact top-20 gallery rows per query (ground truth) |
| `per_query/<setting>_top20.npy` | top-20 per ANN setting (`hnsw_ef*`, `ivfpq_np*`, `lsh_nb*`) |
| `per_query/query_rows.npy`, `gallery_rows.npy` | the fixed split |
| `visual-comparisons/` | same-query grids across methods (from the notebook) |

## Step 4 - Compare in the notebook

```bash
cd experiments/ann-comparison
jupyter lab   # open compare_methods.ipynb, run all cells
```

Sections:

1. one combined metrics table for both datasets,
2. overlaid recall-vs-latency / recall-vs-size plots and parameter sweeps,
3. cross-dataset summary with speed-up and compression vs exact,
4. the same query searched by every method side by side, per dataset,
5. Zappos category-level precision per method,
6. sanity checks.

Sections 1-3 work from the committed `ann_results.json` files alone.
Sections 4-6 need a local benchmark run (for `per_query/`), the
embeddings and the images; anything missing is skipped with a note.

## Results (1,000 queries, 1 search thread)

| Method | Setting | COCO R@10 | COCO mAP@10 | COCO ms/q | COCO MB | Zappos R@10 (= P@10) | Zappos mAP@10 | Zappos ms/q | Zappos MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact FlatIP | - | 1.000 | 1.000 | 2.158 | 358.6 | 1.000 | 1.000 | 0.419 | 95.8 |
| HNSW | efSearch=16 | 0.916 | pending | 0.149 | 390.4 | 0.930 | 0.930 | 0.091 | 108.5 |
| HNSW | efSearch=64 | 0.986 | pending | 0.379 | 390.4 | 0.994 | 0.994 | 0.219 | 108.5 |
| HNSW | efSearch=128 | 0.995 | pending | 0.704 | 390.4 | 0.999 | 0.999 | 0.380 | 108.5 |
| HNSW | efSearch=256 | 0.997 | pending | 1.087 | 390.4 | 1.000 | 1.000 | 0.740 | 108.5 |
| IVF-PQ | nprobe=4 | 0.460 | pending | 0.189 | 8.0 | 0.358 | 0.258 | 0.058 | 2.9 |
| IVF-PQ | nprobe=32 | 0.465 | pending | 0.536 | 8.0 | 0.373 | 0.270 | 0.212 | 2.9 |
| LSH | nbits=256 | 0.324 | pending | 0.215 | 4.5 | 0.229 | 0.147 | 0.080 | 2.0 |
| LSH | nbits=512 | 0.473 | pending | 0.391 | 9.0 | 0.389 | 0.296 | 0.144 | 4.0 |

Full tables: `results/combined_ann_results.csv`.

**Provenance.** The two datasets were measured on different machines, so
compare their speed-ups (in the notebook), not their raw milliseconds:

- **COCO:** from the original run on WSL2 with a 12-core CPU, made before
  Precision@K and mAP@K existed. COCO P@10 equals its R@10 by definition.
  COCO mAP is pending until someone re-runs
  `benchmark_ann.py --dataset coco` with the ViT embeddings in place.
- **Zappos:** re-run on Windows with a 16-core CPU and BLAS pinned to one
  thread.

Key observations:

- **The method ordering is the same on both datasets.** HNSW dominates the
  speed-accuracy trade-off, and IVF-PQ and LSH trade recall for memory.
- **HNSW:** efSearch=64 reaches ~0.99 Recall@10 at 1.9x (Zappos) to 5.7x
  (COCO) the speed of exact search. Its advantage grows with gallery size.
  On the smaller Zappos gallery, efSearch=256 is already slower than a flat
  scan.
- **mAP separates the methods by ranking quality.**
  - HNSW's mAP@10 equals its recall. It scores candidates with exact
    distances, so the neighbours it finds come back in the correct order.
    Any miss is replaced by a less similar item that sorts to the bottom.
  - IVF-PQ and LSH rank by approximate distances (PQ codes, Hamming
    codes), so even the neighbours they do find are often misordered. On
    Zappos, IVF-PQ drops from 0.373 recall to 0.270 mAP@10, and LSH-512
    from 0.389 to 0.296.
- **IVF-PQ:** recall saturates around nprobe=4 (0.465 COCO, 0.373 Zappos)
  because PQ compression distortion is the limit, not missed clusters. In
  return it gives the smallest indexes, 33-45x smaller than exact.
- **Zappos categories:** category precision@10 degrades much more gently
  than vector recall. Exact reaches 0.901 and LSH 0.87, so coarse category
  structure survives approximate search.

## Notes

- Vectors are L2-normalized, so `IndexFlatIP` computes cosine similarity
  directly; `IndexLSH` ranks by L2, which equals cosine ranking on
  normalized vectors.
- Raw latencies are not comparable across datasets (different gallery size,
  dimension and model). Use the notebook's speed-up column instead.
- Dataset links are conveniences for this working repository only. The
  final project submission ZIP must contain no datasets, models or
  cloud-hosted links (course requirement).
