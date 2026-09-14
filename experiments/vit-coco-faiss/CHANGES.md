# Change Log - experiments/vit-coco-faiss

All changes live inside `experiments/vit-coco-faiss/`. No existing team
files (notebook, root requirements, .gitignore, resnet18-zappos, CIRCO
annotations) were modified, and nothing has been committed or pushed.

## 2026-09-14 - Initial experiment setup (WeiW0917)

### Added `unzip_coco.py`
- New file. Multi-threaded extraction of `unlabeled2017.zip` into
  `data/circo/unlabeled2017/`.
- Why: no `unzip` binary on WSL; Python stdlib `zipfile` used instead.
- Resume support: existing files with matching size are skipped, so the
  script can be re-run after interruption.
- Verified: full extraction of 123,403 images completed, sizes match zip
  entries.

### Added `extract_embeddings.py`
- New file. Adapted from `ViT_Embedding.ipynb` (same model
  `vit_b_16`, same `ViT_B_16_Weights.DEFAULT`, same
  `weights.transforms()` preprocessing, same `model.heads = Identity`
  modification).
- Changes relative to the notebook:
  1. Batched inference (default batch 8) instead of one image at a time.
  2. Deterministic order: images sorted by filename.
  3. Output as binary `.npy` memmap instead of CSV. A 123,403 x 768
     float32 matrix is about 362 MiB, versus a multi-GB text CSV.
  4. Vectors L2-normalized before saving, so FAISS `IndexFlatIP` equals
     cosine similarity without re-normalization. (The notebook saved
     unnormalized vectors.)
  5. Resumable: `rows_done` in the metadata JSON; re-running continues
     from the last completed row when source and count match.
  6. Can read images directly from the zip (`--zip`) or from the
     extracted folder (`--images-dir`). The notebook required extracted
     images directly inside `data/circo/`, which does not match the
     zip's `unlabeled2017/` subfolder.
- The notebook itself was not modified.

### Added `validate_embeddings.py`
- New file. Sanity checks before FAISS: shape, dtype, id uniqueness,
  finite values, L2 normalization, metadata consistency, and optional
  id match against the image source.

### Added `exact_search.py`
- New file. FAISS `IndexFlatIP` baseline on the ViT vectors:
  - fixed query set (seed 42) shared by future HNSW / IVF-PQ runs
  - query image excluded from its own top-K
  - FAISS results cross-checked against plain NumPy dot product
  - saves ground-truth top-K rows, summary metrics and readable
    predictions JSON

### Verification performed (2026-09-14)
- Zip extraction: 123,403 images extracted in 2.9 min, verified complete.
- Pilot run: 1,000 images at 74.5 img/s (batch 8, fp32), 1.56 GB VRAM.
- Full run: 123,403 images at 86.2 img/s, about 24 min total, output
  `(123403, 768)` float32, 362 MB.
- `validate_embeddings.py`: all 13 checks PASS, ids match all source
  images.
- `exact_search.py`: 200 queries, K=10, latency 2.22 ms/query,
  451.5 QPS, index 361.5 MB, NumPy cross-check passed.
- NSCC was not needed; local GPU completed the full pipeline.

### Added experiment-local `requirements.txt`
- New file. Root `requirements.txt` intentionally untouched.

### Added `README.md`
- New file. Full usage instructions and pipeline description.

## 2026-09-14 (later) - Visualisation notebook (WeiW0917)

### Added `visualise_results.ipynb`
- New file. Jupyter notebook for inspecting exact-search results:
  - shows the query image and its ViT vector (shape, first values, norm)
  - recalculates cosine scores as dot products (vectors are normalized)
  - displays a query + top-10 result grid with rank, image id and score
  - saves grids to `results/visual-checks/query_<id>_top10.png`
  - sanity checks: query excluded from own results, ids/files exist,
    scores descending, result count correct
- Designed for easy teammate replication: paths resolve automatically from
  either the repo root or the notebook folder, and any query id from
  `predictions.json` can be dropped into `QUERY_IDS`.
- Does not modify `exact_search.py`; scores are recomputed in the notebook
  from the saved embeddings.
- Verified: executed headlessly with `jupyter nbconvert --execute`; all
  checks passed for 3 queries; 3 grids saved.

### Updated experiment-local `requirements.txt`
- Added `matplotlib` and `jupyterlab` (needed by the notebook only).
  Root `requirements.txt` untouched.

### Updated `README.md`
- Added "Step 5 - Visualise the results (Jupyter notebook)".

## 2026-09-14 (latest) - ANN benchmark and comparison notebook (WeiW0917)

### Added `benchmark_ann.py`
- New file. Benchmarks Exact FlatIP, HNSW, IVF-PQ and FAISS IndexLSH on
  the real ViT COCO embeddings.
- Setup: 1,000 queries held out (seed 42), gallery = remaining 122,403
  vectors. This differs from `exact_search.py` (200 queries drawn from
  the indexed gallery) - that earlier run remains as a pipeline test;
  `exact_search.py` was not modified.
- Sweeps: HNSW efSearch {16,32,64,128,256} at M=32/efConstruction=200;
  IVF-PQ nprobe {1,4,8,16,32,64} at nlist=256/m=48/nbits=8, trained on a
  50k gallery sample; LSH nbits {256,512}.
- Saves aggregate metrics plus per-query top-20 for every setting
  (`per_query/*.npy`) so any query can be inspected across methods.
- Timing: searches single-threaded (median of 5 repeats); builds
  multi-threaded. Both thread counts recorded in metadata.
- LSH note: FAISS `IndexLSH` ranks by Hamming distance then refines with
  L2; on L2-normalized vectors L2 ranking equals cosine ranking, so the
  comparison is valid. FAISS LSH is Hamming-based, unlike textbook
  random-hyperplane LSH discussed in lectures - to be stated in the
  report.

### Added `compare_methods.ipynb`
- New file. Interactive comparison notebook:
  - aggregate metrics table (recall@5/10/20, latency, QPS, build, size)
  - recall-vs-latency and recall-vs-size scatter plots
  - per-method parameter sweep curves
  - same-query grids across methods with OK/MISS markers against the
    exact top-10 and per-query recall in each row label
  - per-query recall table across every benchmarked setting
  - sanity checks (shapes, exact recall = 1.0, held-out split verified)
- Adjustable: `QUERY_IDS`, `SHOW_SETTINGS`, `K_DISPLAY`. Reads only the
  benchmark outputs - never recomputes FAISS or ViT.

### Updated `README.md`
- Added "Step 6 - ANN benchmark" with outputs, notebook usage and the
  measured results table.

### Verification performed
- `benchmark_ann.py` ran end-to-end on the real embeddings (runtime about
  6 minutes: exact 4s build, HNSW 37s build, IVF-PQ 25s train+add, LSH ~1s).
- `compare_methods.ipynb` executed headlessly: no errors, all sanity
  checks passed, 3 comparison grids saved to `results/ann/visual-comparisons/`.
- Headline numbers: HNSW efSearch=128 reaches Recall@10 = 0.995 at
  0.70 ms/query (exact: 2.16 ms/query); IVF-PQ saturates at ~0.465
  recall with an 8 MB index (45x smaller than exact).

## 2026-09-14 (final) - Documentation for sharing (WeiW0917)

### Updated `README.md`
- Step 1 now includes the COCO download URL and disk/time expectations.
- Step 2 restructured into two options: download the pre-generated
  embeddings from a shared Google Drive folder, or generate them locally
  with `extract_embeddings.py` (commands unchanged).
- Clarified that the extracted images are only needed for generation and
  the visualisation notebooks; the benchmarks only need the embeddings.
- Added a note that the Drive link is a working-repo convenience only -
  the final submission ZIP must not contain datasets, models or cloud
  links (course requirement).

### Added `data/circo/.gitignore` (one line: `*.npy`)
- New file at the repo's `data/circo/` level; does not modify the root
  `.gitignore`. Prevents anyone from accidentally staging the 362 MB
  embedding files with a `git add .`.

### Notes for the team
- `data/**/*.npy` is not covered by the root `.gitignore` (which ignores
  `data/**/*.csv` and image files). Recommend adding it before anyone
  stages data files.
- Embedding vectors and images stay local; nothing large is committed.
