"""Compare embedding robustness across image perturbations."""

import argparse
import json
from pathlib import Path

from data_loading import (
    DEFAULT_DATA_ROOT,
    discover_datasets,
    format_datasets,
    load_from_dir,
    load_from_zip,
    resolve_dataset,
)
from embedding_models import get_device, run_model
from perturbations import TRANSFORMS, build_variants
from reporting import group_summary, save_curves, save_grid
from menu_choices import choose_from_list, choose_multiple, ask_int, ask_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ResNet18 vs ViT-B/16 embedding robustness to perturbations."
    )
    parser.add_argument("--dataset", help="Discovered dataset name, shown by --list-datasets")
    parser.add_argument("--list-datasets", action="store_true", help="List datasets under data/")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--images-dir", default=None, help="Folder of JPG/PNG images")
    parser.add_argument("--zip", default=None, help="ZIP archive of JPG images")
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--transforms", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prefix", default="robustness")
    parser.add_argument("--use-defaults", action="store_true", help="Skip selection prompts and use the default dataset/model/transform configuration.",)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = discover_datasets(args.data_root)
    if args.list_datasets:
        print(f"Datasets under {args.data_root}:")
        print(format_datasets(datasets))
        return

    if args.use_defaults:
        if not datasets:
            raise RuntimeError(f"No datasets found under {args.data_root}.")
        args.dataset = sorted(datasets)[0]
        args.models = args.models or "resnet18,vit"
        args.transforms = args.transforms or ",".join(t["name"] for t in TRANSFORMS)
        args.num_queries = args.num_queries if args.num_queries is not None else 8
        args.seed = args.seed if args.seed is not None else 42
        args.batch_size = args.batch_size if args.batch_size is not None else 32
        args.out_dir = args.out_dir or "experiments/robustness-comparison/results"
    else:
        if not args.dataset and not args.images_dir and not args.zip:
            if not datasets:
                raise RuntimeError(
                    "No datasets found under the data root. "
                    "Provide --images-dir or --zip instead."
                )
            args.dataset = choose_from_list("Select a dataset", sorted(datasets))

        if args.models is None:
            model_options = ["resnet18", "vit", "resnet18,vit"]
            args.models = choose_from_list(
                "Select models to compare",
                model_options,
                default_index=2,
            )

        if args.transforms is None:
            transform_names = [spec["name"] for spec in TRANSFORMS]
            selected = choose_multiple(
                "Select transforms",
                transform_names,
                default_choices=transform_names,
            )
            args.transforms = ",".join(selected)

        if args.num_queries is None:
            args.num_queries = ask_int("Number of queries per dataset", 8)

        if args.seed is None:
            args.seed = ask_int("Random seed", 42)

        if args.batch_size is None:
            args.batch_size = ask_int("Batch size", 32)

        if args.out_dir is None:
            args.out_dir = ask_string(
                "Output directory",
                "experiments/robustness-comparison/results",
            )
        
    if args.dataset:
        dataset_path = resolve_dataset(args.dataset, args.data_root)
        if dataset_path.suffix.lower() == ".zip":
            images, image_ids = load_from_zip(dataset_path, args.num_queries, args.seed)
        else:
            images, image_ids = load_from_dir(dataset_path, args.num_queries, args.seed)
    elif args.zip:
        images, image_ids = load_from_zip(args.zip, args.num_queries, args.seed)
    elif args.images_dir:
        images, image_ids = load_from_dir(args.images_dir, args.num_queries, args.seed)
    else:
        raise RuntimeError(
            "Provide --dataset, --images-dir, or --zip. "
            "Run with --list-datasets to see available datasets."
        )

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    requested = [name.strip() for name in args.transforms.split(",") if name.strip()]
    valid = {spec["name"] for spec in TRANSFORMS}
    unknown = set(requested) - valid
    if unknown:
        raise ValueError(f"Unknown transform(s): {sorted(unknown)}")
    selected = [spec for spec in TRANSFORMS if spec["name"] in requested]

    device = get_device()
    variants = build_variants(images, image_ids, selected)
    print(f"Queries: {len(images)}")
    print(f"Models: {model_names}")
    print(f"Transforms: {[spec['name'] for spec in selected]}")
    print(f"Variants per model: {len(variants)}")
    print(f"Device: {device}")

    all_rows = []
    for model_name in model_names:
        print(f"Running {model_name} ...")
        all_rows.extend(run_model(model_name, variants, device, args.batch_size))

    frame, summary = group_summary(all_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drift_path = out_dir / f"{args.prefix}_drift.csv"
    summary_path = out_dir / f"{args.prefix}_summary.csv"
    curves_path = out_dir / f"{args.prefix}_curves.png"
    grid_path = out_dir / f"{args.prefix}_grid.png"
    params_path = out_dir / f"{args.prefix}_params.json"

    frame.to_csv(drift_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_curves(summary, model_names, selected, curves_path)
    save_grid(images, image_ids, selected, grid_path)

    params = {
        "models": model_names,
        "transforms": [spec["name"] for spec in selected],
        "num_queries": len(images),
        "seed": args.seed,
        "device": str(device),
        "image_ids": image_ids,
    }
    params_path.write_text(json.dumps(params, indent=2))

    print("\nMean cosine with original embedding:")
    display = summary.copy()
    display["mean"] = display["mean"].round(4)
    print(display.pivot_table(index="transform", columns="model", values="mean").to_string())
    print(f"\nRaw drift:  {drift_path}")
    print(f"Summary:    {summary_path}")
    print(f"Curves:     {curves_path}")
    print(f"Grid:       {grid_path}")
    print(f"Params:     {params_path}")


if __name__ == "__main__":
    main()
