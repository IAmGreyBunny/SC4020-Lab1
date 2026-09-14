# vit-coco-faiss

ViT embeddings on the COCO 2017 unlabeled image collection, plus a FAISS
search comparison: Exact FlatIP vs HNSW vs IVF-PQ vs LSH, with parameter
sweeps, Recall@K against exact ground truth, timing, and index-size
measurements, plus interactive visualisation notebooks.

All files here are new additions for this experiment. No existing team
files were modified. See `CHANGES.md` for the change log.

## Pipeline

```text
COCO unlabeled2017 images
    -> ViT-B/16 (pretrained, classifier head removed)  [extract_embeddings.py]
    -> 768-d L2-normalized vectors (.npy memmap)
    -> Exact FlatIP / HNSW / IVF-PQ / LSH               [benchmark_ann.py]
    -> Recall@5/10/20, latency, QPS, build time, index size
```

## Setup

Python dependencies for this experiment (root `requirements.txt` is
untouched):

```bash
python3 -m pip install -r experiments/vit-coco-faiss/requirements.txt
```

For NVIDIA GPU support install PyTorch with the CUDA build appropriate for
your machine instead of the default wheel.

## Step 1 - Extract the COCO images

Download `unlabeled2017.zip` (about 19 GiB) from
`http://images.cocodataset.org/zips/unlabeled2017.zip` and extract it into
the shared team data folder. Images end up as `data/circo/unlabeled2017/*.jpg`:

```bash
python3 experiments/vit-coco-faiss/unzip_coco.py \
  --zip /path/to/unlabeled2017.zip \
  --dest data/circo \
  --workers 6
```

The extractor is multi-threaded and skips already-extracted files, so it can
be re-run to resume after an interruption.

Disk and time expectations: about 19 GiB for the zip plus 19 GiB extracted
and roughly 3 minutes for extraction. The extracted images are only needed
for Option B below and for the visualisation notebooks; the benchmarks
themselves only need the embedding files.

## Step 2 - Obtain the ViT embeddings

Two options. Both end with the same three files in `data/circo/`.

### Option A - Download the pre-generated embeddings (~363 MB total)

Shared folder: https://drive.google.com/drive/folders/1foFIG1HCD4laqZaU4qaxaTfZBCMrjZqY?usp=drive_link

Download these three files into `data/circo/`:

- `vit_embeddings.npy` - float32 matrix `(123403, 768)`, L2-normalized
- `vit_image_ids.npy` - COCO image ids, row-aligned with the matrix
- `vit_metadata.json` - model and generation details

Then run Step 3 to validate the downloaded files.

### Option B - Generate them yourself (~24 min on a GPU)

Reads images either from the extracted folder or directly from the zip:

```bash
# full run (about 123k images)
python3 experiments/vit-coco-faiss/extract_embeddings.py \
  --images-dir data/circo \
  --output-dir data/circo \
  --prefix vit \
  --batch-size 8

# or directly from the zip, e.g. for a quick pilot
python3 experiments/vit-coco-faiss/extract_embeddings.py \
  --zip /path/to/unlabeled2017.zip \
  --output-dir experiments/vit-coco-faiss/results/pilot \
  --prefix pilot \
  --limit 1000 \
  --batch-size 8
```

Outputs (prefix-based):

| File | Contents |
|---|---|
| `<prefix>_embeddings.npy` | float32 matrix, shape `(N, 768)`, L2-normalized |
| `<prefix>_image_ids.npy` | int64 COCO image ids, row-aligned with the matrix |
| `<prefix>_metadata.json` | model, weights, source, progress (`rows_done`) |

The run is resumable: if interrupted, re-running the same command
continues from the last completed row.

Model details (identical to `ViT_Embedding.ipynb`):
`torchvision vit_b_16`, `ViT_B_16_Weights.DEFAULT`, classifier head
replaced by `Identity`, preprocessing from `weights.transforms()`.

## Step 3 - Validate the embeddings

```bash
python3 experiments/vit-coco-faiss/validate_embeddings.py \
  --emb data/circo/vit_embeddings.npy \
  --ids data/circo/vit_image_ids.npy \
  --meta data/circo/vit_metadata.json \
  --expect-count 123403 \
  --images-dir data/circo
```

Checks: shape, dtype, uniqueness of ids, finite values, normalization,
metadata consistency and id match against the image folder.

## Step 4 - FAISS exact-search baseline

```bash
python3 experiments/vit-coco-faiss/exact_search.py \
  --emb data/circo/vit_embeddings.npy \
  --ids data/circo/vit_image_ids.npy \
  --num-queries 200 \
  --k 10 \
  --seed 42 \
  --threads 1 \
  --out-dir experiments/vit-coco-faiss/results/exact
```

Outputs in `results/exact/`:

| File | Contents |
|---|---|
| `summary.json` | build time, latency, QPS, index size |
| `ground_truth_rows.npy` | exact top-K gallery rows per query (answer key for HNSW / IVF-PQ recall) |
| `query_rows.npy` | the fixed query row indices (seed 42) |
| `predictions.json` | readable query image id -> top-K image ids |

The query image itself is excluded from its own results. FAISS results are
cross-checked against a plain NumPy dot-product search on several queries.

## Step 5 - Visualise the results (Jupyter notebook)

```bash
cd experiments/vit-coco-faiss
jupyter lab
# open visualise_results.ipynb and run all cells
```

Or run it headlessly:

```bash
jupyter nbconvert --to notebook --execute \
  --output /tmp/vis.ipynb \
  experiments/vit-coco-faiss/visualise_results.ipynb
```

The notebook shows, for each query id in `QUERY_IDS`:

- the query image and its ViT vector (shape, values, norm)
- the top-10 retrieved images with rank, image id and cosine score
- sanity checks (no self-result, files exist, scores descending)

To visualise other queries, copy any key from `results/exact/predictions.json`
into `QUERY_IDS` and re-run. It only needs the `.npy` embeddings, the ids,
`predictions.json` and the COCO images - it does not re-run ViT or FAISS.

Result grids are saved to `results/visual-checks/query_<id>_top10.png`.

## Step 6 - ANN benchmark (HNSW / IVF-PQ / LSH vs Exact)

```bash
python3 experiments/vit-coco-faiss/benchmark_ann.py
```

Setup: 1,000 query vectors held out (seed 42); the remaining 122,403
vectors form the gallery. IVF-PQ trains on a 50k gallery sample only.
Exact search provides the top-20 ground truth. All searches are
single-threaded; index builds use all cores.

Outputs in `results/ann/`:

| File | Contents |
|---|---|
| `ann_results.json` / `.csv` | per method/setting: Recall@5/10/20, latency, QPS, build time, index size |
| `per_query/exact_top20.npy` | exact top-20 gallery rows per query (ground truth) |
| `per_query/<setting>_top20.npy` | top-20 per ANN setting (`hnsw_ef*`, `ivfpq_np*`, `lsh_nb*`) |
| `per_query/query_rows.npy`, `gallery_rows.npy` | the fixed split |
| `visual-comparisons/` | same-query grids across methods (from the notebook) |

### Compare methods interactively

```bash
cd experiments/vit-coco-faiss
jupyter lab   # open compare_methods.ipynb, run all cells
```

`compare_methods.ipynb` shows the aggregate table, recall-vs-latency and
recall-vs-size plots, parameter sweep curves, and **the same query
searched by every method side by side** with OK/MISS markers against the
exact top-10. Adjust `QUERY_IDS` and `SHOW_SETTINGS` in the config cell.

### ANN results (123,403 gallery, 1,000 queries, 768-d, 1 thread)

| Method | Setting | Recall@10 | ms/query | QPS | Build (s) | Index (MB) |
|---|---|---:|---:|---:|---:|---:|
| Exact FlatIP | - | 1.000 | 2.158 | 463 | 4.0 | 358.6 |
| HNSW | efSearch=16 | 0.916 | 0.149 | 6,699 | 36.9 | 390.4 |
| HNSW | efSearch=32 | 0.963 | 0.236 | 4,240 | 36.9 | 390.4 |
| HNSW | efSearch=64 | 0.986 | 0.379 | 2,635 | 36.9 | 390.4 |
| HNSW | efSearch=128 | 0.995 | 0.704 | 1,421 | 36.9 | 390.4 |
| HNSW | efSearch=256 | 0.997 | 1.087 | 920 | 36.9 | 390.4 |
| IVF-PQ | nprobe=1 | 0.403 | 0.105 | 9,486 | 25.4 | 8.0 |
| IVF-PQ | nprobe=4 | 0.460 | 0.189 | 5,304 | 25.4 | 8.0 |
| IVF-PQ | nprobe=32 | 0.465 | 0.536 | 1,865 | 25.4 | 8.0 |
| IVF-PQ | nprobe=64 | 0.465 | 1.066 | 938 | 25.4 | 8.0 |
| LSH | nbits=256 | 0.324 | 0.215 | 4,659 | 1.0 | 4.5 |
| LSH | nbits=512 | 0.473 | 0.391 | 2,559 | 1.3 | 9.0 |

Key observations:

- HNSW dominates the speed-accuracy trade-off: efSearch=128 reaches 0.995
  Recall@10 at 3x the speed of exact search; efSearch=16 is 14x faster at
  0.916 recall.
- IVF-PQ recall saturates around 0.465: beyond nprobe=4 the limiting
  factor is PQ compression distortion (48 bytes/vector), not missed
  clusters. Its index is 45x smaller than exact (8 MB vs 359 MB).
- FAISS IndexLSH (Hamming-based) trails HNSW clearly at this scale,
  consistent with it being an older technique.

## Notes and conventions

- Vectors are L2-normalized at save time, so FAISS `IndexFlatIP` computes
  cosine similarity directly.
- Rows are ordered by sorted filename, making runs deterministic and
  comparable across machines.
- Timing runs use a single FAISS thread by default for reproducibility.
- Large outputs (`data/**/*.npy`) are generated locally and are not
  committed.
- The team `.gitignore` currently ignores `data/**/*.csv` and images but
  **not** `data/**/*.npy`; keep that in mind before staging files.
- The Google Drive link above is a convenience for this working repository
  only. The final project submission ZIP must contain no datasets, models
  or cloud-hosted links (course requirement).

## Results so far

Measured on WSL2, RTX 3050 Laptop GPU (4 GB), 12-core CPU, torch 2.14
CUDA build, fp32, single FAISS thread.

### Embedding generation

| Stage | Result |
|---|---|
| Zip extraction (123,403 images, 6 threads) | 2.9 min |
| Pilot (1,000 images, batch 8, fp32) | 74.5 img/s, 1.56 GB VRAM |
| Full run (123,403 images, batch 8, fp32) | 86.2 img/s, about 24 min total |
| Output | `(123403, 768)` float32, 362 MB |

Validation: all checks passed (shape, dtype, unique ids, finite values,
L2 norms = 1.0, ids match all 123,403 source images).

### FAISS exact search baseline (IndexFlatIP)

| Metric | Value |
|---|---|
| Dataset | 123,403 vectors, 768-d |
| Queries | 200 (seed 42), K=10, self excluded |
| Index build | 1.18 s |
| Latency | 2.22 ms/query |
| Throughput | 451.5 QPS |
| Index size | 361.5 MB |

FAISS results were cross-checked against plain NumPy dot-product search.
The saved `ground_truth_rows.npy` is the answer key for the upcoming
HNSW and IVF-PQ recall comparisons.
