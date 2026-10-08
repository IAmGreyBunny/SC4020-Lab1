# ResNet18 + UT-Zap50K embeddings

This experiment generates ResNet18 image embeddings for the UT-Zap50K shoe
dataset. Its handoff format matches the ViT + COCO embedding pipeline:

| Model | Embedding file | Image-ID file | Shape and dtype |
|---|---|---|---|
| ViT + COCO | `data/circo/vit_embeddings.npy` | `data/circo/vit_image_ids.npy` | `(N, 768)` `float32`; `(N,)` `int64` |
| ResNet18 + Zappos | `data/zappos/resnet_embeddings.npy` | `data/zappos/resnet_image_ids.npy` | `(N, 512)` `float32`; `(N,)` `int64` |

The generated arrays feed the FAISS ANN benchmark in this folder
(`benchmark_ann.py` + `compare_methods.ipynb`), which applies the same
Exact / HNSW / IVF-PQ / LSH protocol as the ViT-COCO experiment
(`experiments/vit-coco-faiss/`) to this second dataset.

## Pipeline

```text
UT-Zap50K JPG images
    -> pretrained torchvision ResNet18
    -> classification layer replaced by Identity
    -> 512-dimensional avgpool features
    -> float32 L2 normalization
    -> resnet_embeddings.npy + resnet_image_ids.npy
    -> Exact FlatIP / HNSW / IVF-PQ / LSH      [benchmark_ann.py]
    -> Recall@5/10/20, latency, QPS, build time, index size
```

## Files

```text
experiments/resnet18-zappos/
├── README.md
├── common.py                 # Zappos path discovery and deterministic ordering
├── extract_features.py       # required embedding-generation pipeline
├── requirements.txt          # Python dependencies
├── search.py                 # earlier retrieval experiment; not required for handoff
├── evaluate.py               # earlier label evaluation; not required for handoff
├── benchmark_ann.py          # FAISS ANN benchmark (Exact/HNSW/IVF-PQ/LSH)
├── compare_methods.ipynb     # aggregate plots + same-query method comparison
└── results/
    ├── evaluation_summary_N50025.csv     # earlier label evaluation
    ├── search_N50025_K20_Q0.png          # earlier single-query result
    └── ann/                              # ANN benchmark outputs
        ├── ann_results.json / ann_results.csv
        ├── per_query/                    # generated locally; not committed
        └── visual-comparisons/           # saved grids from the notebook
```

For the embedding handoff, only `common.py`, `extract_features.py`, and the
installed dependencies are required.

## Dataset location

The dataset is stored outside this repository and is not copied or committed.
The current default is configured near the top of `common.py`:

```text
/Users/kohjiaxin/Y4S1/ResNet18/image-similarity-search/data/zappos/ut-zap50k-images/ut-zap50k-images
```

A different location can be supplied at runtime with `--images-dir`, without
editing the source code.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/resnet18-zappos/requirements.txt
```

## Generate the full embeddings

Run from the repository root:

```bash
python experiments/resnet18-zappos/extract_features.py
```

The default settings are:

- external dataset path shown above
- all 50,025 JPG images
- batch size 32
- output directory `data/zappos`
- output prefix `resnet`
- CUDA, Apple MPS, or CPU selected automatically

The two handoff files are:

```text
data/zappos/resnet_embeddings.npy
data/zappos/resnet_image_ids.npy
```

## Configurable command

The command-line interface follows the ViT extractor's main conventions:

```bash
python experiments/resnet18-zappos/extract_features.py \
  --images-dir /path/to/ut-zap50k-images \
  --output-dir data/zappos \
  --prefix resnet \
  --batch-size 32
```

For a quick smoke test, add a limit and use a temporary prefix or directory so
the final arrays are not overwritten:

```bash
python experiments/resnet18-zappos/extract_features.py \
  --limit 1000 \
  --output-dir /tmp/resnet-zappos-pilot \
  --prefix resnet
```

Omitting `--limit` processes the full available dataset.

## Output contract

`resnet_embeddings.npy` contains:

- shape `(N, 512)`
- dtype `float32`
- one L2-normalized ResNet18 vector per row

`resnet_image_ids.npy` contains:

- shape `(N,)`
- dtype `int64`
- one ID for each embedding row

The arrays are row-aligned:

```python
resnet_embeddings[i]  # embedding for image ID resnet_image_ids[i]
```

UT-Zap50K filenames do not expose a convenient COCO-style integer ID. Images
are therefore sorted lexicographically by full path and assigned sequential IDs
`0 ... N-1`. The mapping is stable when the dataset contents are unchanged.

## Built-in validation

After writing the arrays, the extractor reloads them and prints:

```text
Embeddings: (50025, 512)
Image IDs: (50025,)
Embedding dtype: float32
Image ID dtype: int64
Row counts match: True
L2 normalized: True
```

The current generated full arrays have been checked and pass this contract.
Generated `.npy` files are ignored by Git and should be transferred separately
to the teammate who will build the indexes.

## Single-query retrieval test of the handoff files

`search.py` reads the exact two `.npy` handoff files and searches one query
against all 50,025 embeddings. With no `--query-id`, seed 42 reproducibly
selects the nonzero query ID 4465:

```bash
python experiments/resnet18-zappos/search.py
```

The saved results are:

```text
experiments/resnet18-zappos/results/single_query_summary_N50025_Q4465.csv
experiments/resnet18-zappos/results/single_query_top20_N50025_Q4465.png
```

Query 4465 is an ankle boot. Its results are:

| K | Category precision | Category recall | Subcategory precision | Subcategory recall |
|---:|---:|---:|---:|---:|
| 5 | 80% | 0.0312% | 80% | 0.0683% |
| 10 | 90% | 0.0701% | 90% | 0.1537% |
| 20 | 95% | 0.1481% | 95% | 0.3246% |

Choose a particular deterministic ID with, for example, `--query-id 12345`.
The PNG uses a noninteractive backend, so it is saved without blocking the
terminal with a plot window.

## FAISS ANN benchmark on this dataset (second dataset)

The same protocol as the ViT-COCO benchmark, applied to the ResNet18
embeddings: 1,000 held-out query vectors (seed 42), the remaining 49,025
vectors form the gallery, exact FlatIP produces the top-20 ground truth,
and HNSW / IVF-PQ / LSH are evaluated with Recall@5/10/20, latency, QPS,
build time and index size. Searches are single-threaded for
reproducibility; index builds use all cores.

### Step 1 - Obtain the embeddings

**Option A - use the generated handoff files** (about 98 MB + 0.4 MB):
place `resnet_embeddings.npy` and `resnet_image_ids.npy` into
`data/zappos/`. They are shared separately (not committed; ignored by
`data/zappos/.gitignore`).

**Option B - generate them yourself** with `extract_features.py` as
described above (`--images-dir` pointing at your local UT-Zap50K copy).

### Step 2 - Obtain the UT-Zap50K images (notebook only)

The benchmark itself needs only the embeddings. The visual notebook
additionally needs the images at
`data/zappos/ut-zap50k-images/ut-zap50k-images`. If you do not have the
dataset locally, download it from Kaggle (853 MB, 50,025 images,
CC BY-SA 4.0) and move the images subtree into place:

```python
import shutil
from pathlib import Path

import kagglehub

root = Path(kagglehub.dataset_download("aryashah2k/large-shoe-dataset-ut-zappos50k"))
src = root / "ut-zap50k-images" / "ut-zap50k-images"
dst = Path("data/zappos/ut-zap50k-images/ut-zap50k-images")
dst.parent.mkdir(parents=True, exist_ok=True)
if not dst.exists():
    shutil.move(str(src), str(dst))
```

Only the non-square `ut-zap50k-images` tree is needed (the archive also
contains square images, metadata and precomputed features, which stay in
the kagglehub cache).

### Step 3 - Run the benchmark

From the repository root:

```bash
python3 experiments/resnet18-zappos/benchmark_ann.py
```

Two data-forced differences from the COCO benchmark:

- `--pq-m 32` instead of 48: FAISS requires the PQ sub-quantizer count to
  divide the dimension, and 512 % 48 != 0.
- IVF-PQ trains on the full gallery (49,025 rows, below the 50,000 sample
  cap), so its training uses every gallery vector.

Outputs in `results/ann/`:

| File | Contents |
|---|---|
| `ann_results.json` / `.csv` | per method/setting: Recall@5/10/20, latency, QPS, build time, index size |
| `per_query/exact_top20.npy` | exact top-20 gallery rows per query (ground truth) |
| `per_query/<setting>_top20.npy` | top-20 per ANN setting (`hnsw_ef*`, `ivfpq_np*`, `lsh_nb*`) |
| `per_query/query_rows.npy`, `gallery_rows.npy` | the fixed split |
| `visual-comparisons/` | same-query grids across methods (from the notebook) |

### Step 4 - Compare methods interactively

```bash
cd experiments/resnet18-zappos
jupyter lab   # open compare_methods.ipynb, run all cells
```

`compare_methods.ipynb` shows the aggregate table, recall-vs-latency and
recall-vs-size plots, parameter sweep curves, **the same query searched by
every method side by side** with OK/MISS markers against the exact top-10
and UT-Zap50K category labels, category-level precision per method, and a
cross-dataset comparison against the ViT-COCO results.

### Handoff validation

Before benchmarking, the received `.npy` files were validated against the
Kaggle-downloaded images: 50,025 rows, `(50025, 512)` float32, sequential
IDs, unit L2 norms, and the sorted-path ordering reproduced the teammate's
documented query-4465 single-query results exactly (top-10 category
precision 9/10, top-20 19/20) - confirming the row-to-image mapping.

### ANN results (49,025 gallery, 1,000 queries, 512-d, 1 thread)

| Method | Setting | Recall@10 | ms/query | QPS | Build (s) | Index (MB) |
|---|---|---:|---:|---:|---:|---:|
| Exact FlatIP | - | 1.000 | 0.623 | 1,604 | 0.1 | 95.8 |
| HNSW | efSearch=16 | 0.929 | 0.109 | 9,164 | 8.3 | 108.5 |
| HNSW | efSearch=32 | 0.976 | 0.157 | 6,381 | 8.3 | 108.5 |
| HNSW | efSearch=64 | 0.993 | 0.261 | 3,830 | 8.3 | 108.5 |
| HNSW | efSearch=128 | 0.998 | 0.451 | 2,217 | 8.3 | 108.5 |
| HNSW | efSearch=256 | 1.000 | 0.795 | 1,258 | 8.3 | 108.5 |
| IVF-PQ | nprobe=1 | 0.277 | 0.062 | 16,004 | 7.4 | 2.9 |
| IVF-PQ | nprobe=4 | 0.358 | 0.077 | 13,034 | 7.4 | 2.9 |
| IVF-PQ | nprobe=32 | 0.373 | 0.233 | 4,295 | 7.4 | 2.9 |
| IVF-PQ | nprobe=64 | 0.373 | 0.391 | 2,560 | 7.4 | 2.9 |
| LSH | nbits=256 | 0.229 | 0.097 | 10,337 | 0.4 | 2.0 |
| LSH | nbits=512 | 0.389 | 0.174 | 5,752 | 0.7 | 4.0 |

Key observations:

- HNSW reaches **perfect Recall@10 (1.000)** at efSearch=256. Unlike on
  COCO, that setting is slightly slower than exact search - with only
  49,025 512-d vectors a flat scan is already fast, so the graph index
  stops paying off near the top of the accuracy range. The practical sweet
  spot is efSearch=64: 0.993 recall at 2.4x the speed of exact.
- IVF-PQ recall saturates at 0.373: the limiting factor is PQ compression
  distortion (32 bytes/vector), not missed clusters. Its index is 33x
  smaller than exact (2.9 MB vs 95.8 MB).
- Category precision@10 (UT-Zap50K folder labels) degrades much more
  gracefully than vector recall: exact retrieves same-category neighbours
  90.1% of the time, and even LSH keeps 87% - the coarse category
  structure survives approximate search, while the fine-grained ranking
  does not.
- Cross-dataset: the method ordering matches the ViT-COCO benchmark
  (HNSW dominates, then LSH, then IVF-PQ at this compression level), so
  the conclusions generalise across datasets. The notebook's section 4
  shows both datasets side by side.

### Notes

- Vectors are L2-normalized, so FAISS `IndexFlatIP` computes cosine
  similarity directly; FAISS `IndexLSH` ranks by L2, which equals cosine
  ranking on normalized vectors.
- The benchmark and notebook only read the two `.npy` handoff files; they
  never re-run ResNet18.
- `data/zappos/.gitignore` ignores `*.npy` and `*.pkl`, so the large
  arrays cannot be committed accidentally.
- Dataset links (Kaggle) are conveniences for this working repository
  only. The final project submission ZIP must contain no datasets, models
  or cloud-hosted links (course requirement).
