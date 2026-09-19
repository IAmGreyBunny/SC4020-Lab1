"""Aggregation and visual report generation."""


def group_summary(drift_rows):
    import pandas as pd

    frame = pd.DataFrame(drift_rows)
    summary = (
        frame.groupby(["model", "transform", "level"])["cosine"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    return frame, summary


def save_curves(summary, models, transforms, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, spec in zip(axes.ravel(), transforms):
        block = summary[summary["transform"] == spec["name"]]
        plotted = 0
        for model_name in models:
            curve = block[block["model"] == model_name]
            if not curve.empty:
                axis.plot(curve["level"], curve["mean"], marker="o", label=model_name)
                plotted += 1
        axis.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(spec["name"])
        axis.set_xlabel(spec["unit"] or "factor")
        axis.set_ylabel("mean cosine with original")
        axis.set_ylim(-0.05, 1.05)
        if plotted:
            axis.legend()
    figure.suptitle("Embedding drift: ResNet18 vs ViT-B/16", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_grid(images, image_ids, selected, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from perturbations import level_label, perturb

    sample = images[0]
    columns = max(len(spec["values"]) for spec in selected)
    rows = 1 + len(selected)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.5))
    for column in range(columns):
        axes[0][column].imshow(sample)
        axes[0][column].set_title("original", fontsize=9)
        axes[0][column].axis("off")
    for row, spec in enumerate(selected, start=1):
        for column, value in enumerate(spec["values"]):
            axes[row][column].imshow(perturb(sample, spec["name"], value))
            axes[row][column].set_title(level_label(spec, value), fontsize=8)
            axes[row][column].axis("off")
        for column in range(len(spec["values"]), columns):
            axes[row][column].axis("off")
    figure.suptitle(f"Query: {image_ids[0]}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
