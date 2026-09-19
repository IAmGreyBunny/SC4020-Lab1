# robustness-comparison

Compares **ResNet18** vs **ViT-B/16** embedding robustness to common image
perturbations: rotation, crop, brightness, blur, saturation and hue.

The two team experiments use different datasets (ResNet18 on UT-Zap50K, ViT on
COCO). This experiment removes that confound by feeding both models **the same
query images** and measuring how much each perturbation moves the embedding.

## Metric

For query image *q* and perturbation *p*:

```text
cosine = (E(q) . E(p(q))) / (|E(q)| |E(p(q))|)
```

Both models produce L2-normalized embeddings, so this is their cosine
similarity. A value near **1.0** means the perturbation does not change what
the model "sees"; a drop toward 0 means the embedding drifts. Reported per
model, transform and intensity, aggregated over the query sample.

## Pipeline

```text
query images (folder or ZIP)
    -> original + perturbed variants (see table)
    -> ResNet18 embedding (512-d, fc removed)    -> cosine vs original
    -> ViT-B/16 embedding (768-d, head removed)  -> cosine vs original
    -> drift.csv (per variant), summary.csv (per model/transform/level)
    -> curves.png (cosine vs intensity), grid.png (perturbed thumbnails)
```

The models and their ImageNet preprocessing are the same ones used by the team
pipelines (`vt_b_16` with the head replaced by `Identity`, ResNet18 with `fc`
replaced by `Identity`).

## Perturbations

| Transform | Values | Notes |
|---|---|---|
| rotation | 5, 10, 20, 45, 90, 180 deg | counter-clockwise, within the canvas |
| crop | 0.9, 0.8, 0.7, 0.5, 0.3 | center-crop fraction, resized back |
| brightness | 0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 3.0 | multiplier |
| blur | 0.5, 1.0, 2.0, 4.0 px | Gaussian radius |
| saturation | 0.2, 0.4, 0.7, 1.3, 1.7, 2.5 | multiplier |
| hue | -0.3, -0.15, 0.15, 0.3 | fraction of 360 deg |

## Setup

From the repository root (same stack as the other experiments):

```bash
python3 -m pip install -r experiments/robustness-comparison/requirements.txt
```

## Usage

Query images can come from either the zappos ZIP bundled in the repo data or
any folder of images (e.g. the COCO `unlabeled2017` folder).

Run
```bash
python experiments/robustness-comparison/compare_robustness.py
```

Or alternatively execute by CLI

```bash
# list image directories and ZIP archives found under data/
python experiments/robustness-comparison/compare_robustness.py \
  --list-datasets

# both models on a zipped folder of the local zappos archive
python experiments/robustness-comparison/compare_robustness.py \
  --zip data/zappos/archive.zip --num-queries 8

# both models on a unzipped folder of COCO images
python experiments/robustness-comparison/compare_robustness.py \
  --images-dir data/circo/unlabeled2017 --num-queries 8

# ResNet18 only, subset of perturbations
python experiments/robustness-comparison/compare_robustness.py \
  --zip data/zappos/archive.zip --models resnet18 \
  --transforms rotation,blur --num-queries 8
```


## Outputs

Written to `experiments/robustness-comparison/results/` (prefix `robustness`):

| File | Contents |
|---|---|
| `robustness_drift.csv` | one row per query/variant/model: cosine with original |
| `robustness_summary.csv` | mean/std/min/max cosine per model/transform/level |
| `robustness_curves.png` | cosine-vs-intensity curves, one subplot per transform |
| `robustness_grid.png` | thumbnails of a sample query under every perturbation |
| `robustness_params.json` | models, transforms, seed, queries used |

## Results so far

Initial benchmark: CPU, 8 UT-Zap50K queries (seed 42), all six transforms,
both models. Values are mean cosine between original and perturbed embedding
(higher = more robust). Full numbers are in `results/robustness_summary.csv`.

| Transform | ResNet18 | ViT-B/16 |
|---|---:|---:|
| rotation | 0.851 | 0.706 |
| crop | 0.851 | 0.752 |
| brightness | 0.932 | 0.859 |
| blur | 0.852 | 0.754 |
| saturation | 0.987 | 0.977 |
| hue | 0.960 | 0.902 |

ResNet18 drifts less than ViT-B/16 on every perturbation, with the largest gap
for rotation (+0.15). Geometric perturbations (rotation, crop) hurt both models
more than colour-only ones (saturation). Replace this with a larger, balanced
query sample before drawing conclusions for the report.