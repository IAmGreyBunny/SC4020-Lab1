"""Validate generated embedding files before using them in FAISS.

New file for this experiment. Checks shape, dtype, id uniqueness, finite
values, normalization and optional id match against the image source.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{(' - ' + detail) if detail else ''}")
    return condition


def collect_source_ids(args):
    if args.zip:
        with ZipFile(args.zip) as archive:
            names = sorted(
                item.filename
                for item in archive.infolist()
                if item.filename.lower().endswith((".jpg", ".jpeg"))
            )
    else:
        names = sorted(
            str(path)
            for path in Path(args.images_dir).rglob("*")
            if path.suffix.lower() in (".jpg", ".jpeg")
        )
    return [int(Path(name).stem) for name in names[: args.limit or None]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--meta")
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--expect-dim", type=int, default=768)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--zip")
    source.add_argument("--images-dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    ok = True

    emb_path, ids_path = Path(args.emb), Path(args.ids)
    ok &= check("embedding file exists", emb_path.exists())
    ok &= check("ids file exists", ids_path.exists())
    if not ok:
        raise SystemExit(1)

    embeddings = np.load(emb_path, mmap_mode="r")
    ids = np.load(ids_path)

    ok &= check(
        "embeddings are 2-D",
        embeddings.ndim == 2,
        f"shape={embeddings.shape}",
    )
    ok &= check(
        "dtype is float32",
        embeddings.dtype == np.float32,
        f"dtype={embeddings.dtype}",
    )
    ok &= check(
        "dimension matches expectation",
        embeddings.shape[1] == args.expect_dim,
        f"dim={embeddings.shape[1]}",
    )
    ok &= check(
        "row count matches ids",
        embeddings.shape[0] == ids.shape[0],
        f"{embeddings.shape[0]} vs {ids.shape[0]}",
    )

    if args.expect_count is not None:
        ok &= check(
            "row count matches expected count",
            embeddings.shape[0] == args.expect_count,
            f"{embeddings.shape[0]} vs {args.expect_count}",
        )

    ok &= check("ids are unique", len(np.unique(ids)) == len(ids))
    ok &= check("ids are non-negative", bool((ids >= 0).all()))

    # Finite check over the whole matrix (works on a memmap in chunks).
    finite = True
    for start in range(0, embeddings.shape[0], 4096):
        if not np.isfinite(embeddings[start : start + 4096]).all():
            finite = False
            break
    ok &= check("all values are finite", finite)

    # Norms should be ~1.0 if vectors were L2-normalized at save time.
    sample = np.asarray(embeddings[: min(2048, embeddings.shape[0])])
    norms = np.linalg.norm(sample, axis=1)
    ok &= check(
        "vectors are L2-normalized",
        bool(np.allclose(norms, 1.0, atol=1e-3)),
        f"mean norm={norms.mean():.4f}",
    )

    if args.meta:
        meta = json.loads(Path(args.meta).read_text())
        ok &= check(
            "metadata rows_done equals row count",
            meta.get("rows_done") == embeddings.shape[0],
            f"rows_done={meta.get('rows_done')}",
        )

    if args.zip or args.images_dir:
        source_ids = collect_source_ids(args)
        ok &= check(
            "ids match image source",
            list(ids) == source_ids,
            f"{len(ids)} vs {len(source_ids)}",
        )

    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
