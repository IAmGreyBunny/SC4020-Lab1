# SC4020 Lab 1: ANN search comparison

Compares exact search, LSH, IVF-Flat, IVF-PQ, HNSW and HNSW-PQ (FAISS) on image
embeddings from two models (ViT-B/16, ResNet18) on two datasets (COCO unlabeled2017,
UT-Zap50K): four configs, `vit-coco`, `resnet-coco`, `vit-zappos`, `resnet-zappos`.
All code is in `experiments/ann-comparison/`. Run every command from the repository root.

## Setup

```bash
python -m venv venv
venv/Scripts/python -m pip install -r experiments/ann-comparison/requirements.txt
```

For GPU embedding extraction, install the CUDA build of PyTorch instead.

## 1. Images

- **COCO**: put the `unlabeled2017` JPGs in `data/circo/`. From the zip:
  `python experiments/ann-comparison/pipeline/coco/unzip_coco.py --zip /path/to/unlabeled2017.zip --dest data/circo`
- **Zappos**: UT-Zap50K from Kaggle (`aryashah2k/large-shoe-dataset-ut-zappos50k`), placed so that
  `data/zappos/ut-zap50k-images/ut-zap50k-images/{Boots,Sandals,Shoes,Slippers}/` exists.

## 2. Embeddings (GPU recommended)

```bash
python experiments/ann-comparison/pipeline/extract_embeddings.py --model vit    --dataset coco
python experiments/ann-comparison/pipeline/extract_embeddings.py --model resnet --dataset coco
python experiments/ann-comparison/pipeline/extract_embeddings.py --model vit    --dataset zappos
python experiments/ann-comparison/pipeline/extract_embeddings.py --model resnet --dataset zappos
```

## 3. Experiments

Run these in order. Each step reads the results of the steps before it, and every
script takes `--config <name>` to run a single config.

```bash
python experiments/ann-comparison/benchmark_ann.py --config all        # main ANN benchmark
python experiments/ann-comparison/pq_sweep.py --config all             # PQ compression (m) sweep
python experiments/ann-comparison/pipeline/corruption.py --config all --num-queries 200   # corrupted queries, exact search
python experiments/ann-comparison/pipeline/index_robustness.py --config all               # corrupted queries, ANN indexes
```

Results are written to `experiments/ann-comparison/results/`.

## 4. Figures and tables

Open and run all cells of:

- `experiments/ann-comparison/visualize_results.ipynb`: every figure (saved to `media/`) and the best-setting and ranking tables.
- `experiments/ann-comparison/index_robustness.ipynb`: ANN index robustness under query corruption.
