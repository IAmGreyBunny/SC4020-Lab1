# ResNet18 + UT-Zap50K embeddings

This experiment generates ResNet18 image embeddings for the UT-Zap50K shoe
dataset. Its handoff format matches the ViT + COCO embedding pipeline:

| Model | Embedding file | Image-ID file | Shape and dtype |
|---|---|---|---|
| ViT + COCO | `data/circo/vit_embeddings.npy` | `data/circo/vit_image_ids.npy` | `(N, 768)` `float32`; `(N,)` `int64` |
| ResNet18 + Zappos | `data/zappos/resnet_embeddings.npy` | `data/zappos/resnet_image_ids.npy` | `(N, 512)` `float32`; `(N,)` `int64` |

FAISS, HNSW, and other retrieval indexes are outside this experiment's scope.
The two generated arrays are intended to be passed to the teammate responsible
for those comparisons.

## Pipeline

```text
UT-Zap50K JPG images
    -> pretrained torchvision ResNet18
    -> classification layer replaced by Identity
    -> 512-dimensional avgpool features
    -> float32 L2 normalization
    -> resnet_embeddings.npy + resnet_image_ids.npy
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
└── results/                  # earlier experimental results
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
