import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # m2caiSeg evaluation — SAM2 baseline / fine-tuned

    Runs SAM2 inference on the m2caiSeg preprocessed dataset and computes
    IoU / Dice / Precision / Recall per class.

    Layout assumed (produced by `m2caiSeg_PreProcessingMasks.py`):

    ```
    {root}/{dataset}/{split}/Frames/{image_id}.jpg
    {root}/{dataset}/{split}/Masks/{class_name}/{image_id}.png
    ```

    Predictions land at:

    ```
    {root}/{dataset}/{split}/PredictedMasks/{class_name}/{image_id}.png
    ```

    Re-running clears `PredictedMasks/` for the chosen (dataset, split)
    before writing — the notebook is idempotent at the predicted-masks tier.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import random
    import shutil
    import time
    from pathlib import Path

    import altair as alt
    import cv2
    import numpy as np
    import polars as pl
    import torch
    from PIL import Image

    return Image, Path, alt, cv2, np, pl, shutil, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    # Anchor defaults to project root (located by walking up looking for
    # pyproject.toml). __file__ is unreliable inside marimo cells (sometimes a
    # bare filename), so we use a marker file instead. This makes the notebook
    # work no matter what cwd marimo was launched from.
    def _project_root() -> Path:
        for start in (Path.cwd(), Path(__file__).resolve().parent):
            for parent in (start, *start.parents):
                if (parent / "pyproject.toml").is_file():
                    return parent
        return Path.cwd()

    PROJECT_ROOT = _project_root()

    dataset_root = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing"),
        label="Dataset root",
        full_width=True,
    )
    sam_configs_dir = mo.ui.text(
        value=str(PROJECT_ROOT / "sam_configs"),
        label="SAM2 hydra config dir",
        full_width=True,
    )
    eval_results_dir = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing" / "Eval_Results"),
        label="Eval results dir (CSV outputs)",
        full_width=True,
    )

    datasets = mo.ui.multiselect(
        options=["m2caiSeg"], value=["m2caiSeg"], label="Datasets",
    )
    split = mo.ui.radio(
        options=["val", "test", "train"], value="val", label="Split",
    )

    # Model presets: label → (config_name, absolute checkpoint path).
    MODEL_PRESETS = {
        "Hiera base+ (baseline)": (
            "sam2.1/sam2.1_hiera_b+.yaml",
            str(PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
        ),
        "Hiera large (baseline)": (
            "sam2.1/sam2.1_hiera_l.yaml",
            str(PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"),
        ),
        "Fine-tuned Curated400 (Hiera base+)": (
            "sam2.1/sam2.1_hiera_b+.yaml",
            str(PROJECT_ROOT / "Curated400_checkpoint_26.pt"),
        ),
    }
    model_preset = mo.ui.dropdown(
        options=list(MODEL_PRESETS.keys()),
        value="Hiera base+ (baseline)",
        label="Model preset",
    )

    num_points = mo.ui.number(start=1, stop=50, step=1, value=1, label="Points per prompt")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=42, label="Random seed")

    load_button = mo.ui.run_button(label="Load model")
    run_button = mo.ui.run_button(label="Run inference + metrics")
    return (
        MODEL_PRESETS,
        dataset_root,
        datasets,
        eval_results_dir,
        load_button,
        model_preset,
        num_points,
        run_button,
        sam_configs_dir,
        seed,
        split,
    )


@app.cell(hide_code=True)
def _(
    dataset_root,
    datasets,
    eval_results_dir,
    load_button,
    mo,
    model_preset,
    num_points,
    run_button,
    sam_configs_dir,
    seed,
    split,
):
    mo.vstack([
        dataset_root,
        sam_configs_dir,
        eval_results_dir,
        mo.hstack([datasets, split], justify="start"),
        mo.hstack([model_preset, num_points, seed], justify="start"),
        mo.hstack([load_button, run_button], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pure helpers
    """)
    return


@app.cell
def _(cv2, np):
    def select_point_prompt(mask: np.ndarray, num_points: int, rng):
        """Sample foreground points (mask == 255). Returns (points, labels) or (None, None).

        Points are returned as (x, y) in image coordinates, label 1 (foreground).
        """
        coords = np.argwhere(mask == 255)  # (y, x) pairs
        if len(coords) < num_points:
            return None, None
        idx = rng.choice(len(coords), size=num_points, replace=False)
        pts = [(int(coords[i, 1]), int(coords[i, 0])) for i in idx]
        return pts, [1] * num_points

    def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
        """IoU, Dice, Precision, Recall on uint8 masks (>127 = foreground).

        Resizes prediction to GT size if needed (matches original behavior).
        """
        if pred.ndim > 2 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if gt.ndim > 2 and gt.shape[-1] == 1:
            gt = gt.squeeze(-1)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        p = (pred > 127).astype(np.uint8)
        g = (gt > 127).astype(np.uint8)

        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        ps = int(p.sum())
        gs = int(g.sum())

        return {
            "iou": inter / union if union else 0.0,
            "dice": (2 * inter) / (ps + gs) if (ps + gs) else 0.0,
            "precision": inter / ps if ps else 0.0,
            "recall": inter / gs if gs else 0.0,
        }

    return calculate_metrics, select_point_prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discover (frame, mask) pairs

    Walks `{root}/{dataset}/{split}/Masks/{class}/*.png` and pairs each
    mask with `{root}/{dataset}/{split}/Frames/{stem}.jpg`. Drops masks
    that have no matching frame.
    """)
    return


@app.cell
def _(Path, dataset_root, datasets, mo, pl, split):
    def discover_pairs(root: Path, ds_names: list[str], sp: str) -> pl.DataFrame:
        rows = []
        for ds in ds_names:
            split_dir = root / ds / sp
            masks_root = split_dir / "Masks"
            frames_dir = split_dir / "Frames"
            if not masks_root.is_dir() or not frames_dir.is_dir():
                continue
            for class_dir in sorted(masks_root.iterdir()):
                if not class_dir.is_dir():
                    continue
                for mask in sorted(class_dir.iterdir()):
                    if mask.suffix.lower() != ".png":
                        continue
                    frame = frames_dir / f"{mask.stem}.jpg"
                    if not frame.is_file():
                        continue
                    rows.append({
                        "dataset": ds,
                        "split": sp,
                        "class_name": class_dir.name,
                        "image_id": mask.stem,
                        "frame_path": str(frame),
                        "mask_path": str(mask),
                    })
        if not rows:
            return pl.DataFrame(schema={
                "dataset": pl.String, "split": pl.String, "class_name": pl.String,
                "image_id": pl.String, "frame_path": pl.String, "mask_path": pl.String,
            })
        return pl.DataFrame(rows)

    root = Path(dataset_root.value).expanduser().resolve()
    pairs_df = discover_pairs(root, list(datasets.value), split.value)

    summary = (
        pairs_df.group_by(["dataset", "class_name"])
        .len(name="n_pairs")
        .sort(["dataset", "class_name"])
        if pairs_df.height
        else pl.DataFrame(schema={"dataset": pl.String, "class_name": pl.String, "n_pairs": pl.UInt32})
    )

    mo.vstack([
        mo.md(
            f"**Root**: `{root}`  \n"
            f"**Pairs discovered**: {pairs_df.height}"
        ),
        mo.ui.table(summary, page_size=30),
    ])
    return pairs_df, root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load model

    Press **Load model** to (re)build the SAM2 predictor with the selected
    preset. The cell re-runs whenever the preset changes and the load
    button is still latched on.
    """)
    return


@app.cell
def _(
    MODEL_PRESETS,
    Path,
    load_button,
    mo,
    model_preset,
    sam_configs_dir,
    torch,
):
    mo.stop(not load_button.value, mo.md("_Press **Load model** to build the predictor._"))

    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config_name, ckpt_rel = MODEL_PRESETS[model_preset.value]
    ckpt_path = Path(ckpt_rel).expanduser().resolve()
    cfg_dir = Path(sam_configs_dir.value).expanduser().resolve()

    if not ckpt_path.is_file():
        mo.stop(True, mo.md(f"❌ Checkpoint not found: `{ckpt_path}`"))
    if not cfg_dir.is_dir():
        mo.stop(True, mo.md(f"❌ SAM2 config dir not found: `{cfg_dir}`"))

    # Hydra has global state — reset before re-initialising so model swaps work.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(cfg_dir), version_base=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(config_file=config_name, ckpt_path=str(ckpt_path), device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    mo.md(
        f"**Model loaded**: {model_preset.value}  \n"
        f"**Device**: `{device}`  \n"
        f"**Checkpoint**: `{ckpt_path}`  \n"
        f"**Config**: `{cfg_dir}/{config_name}`"
    )
    return (predictor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run inference

    For each (frame, mask) pair, samples `num_points` foreground points
    from the GT mask, calls SAM2, and writes a binary predicted mask to
    `PredictedMasks/{class}/{image_id}.png`. The predicted-masks tree for
    the chosen (dataset, split) is cleared before the run.
    """)
    return


@app.cell
def _(
    Image,
    cv2,
    datasets,
    mo,
    np,
    num_points,
    pairs_df,
    predictor,
    root,
    run_button,
    seed,
    select_point_prompt,
    shutil,
    split,
    time,
    torch,
):
    mo.stop(not run_button.value, mo.md("_Press **Run inference + metrics** to start._"))
    mo.stop(pairs_df.height == 0, mo.md("_No pairs discovered — check dataset root._"))

    def _run_inference() -> tuple[int, list[str], float]:
        # Idempotent reset of PredictedMasks for the chosen (dataset, split).
        for ds in datasets.value:
            pmask_dir = root / ds / split.value / "PredictedMasks"
            if pmask_dir.exists():
                shutil.rmtree(pmask_dir)

        rng = np.random.default_rng(seed.value)
        skipped: list[str] = []
        n_done = 0
        t_start = time.time()

        for r in pairs_df.iter_rows(named=True):
            with Image.open(r["frame_path"]) as fi:
                frame_rgb = np.array(fi.convert("RGB"))
            with Image.open(r["mask_path"]) as mi:
                gt_mask = np.array(mi.convert("L"))

            pts, labels = select_point_prompt(gt_mask, num_points.value, rng)
            if pts is None:
                skipped.append(f"{r['class_name']}/{r['image_id']}")
                continue

            predictor.set_image(frame_rgb)
            with torch.no_grad():
                masks_out, _, _ = predictor.predict(
                    point_coords=np.array(pts),
                    point_labels=np.array(labels),
                    multimask_output=False,
                )
            binary = (masks_out[0] > 0.5).astype(np.uint8) * 255

            out_path = (
                root / r["dataset"] / r["split"]
                / "PredictedMasks" / r["class_name"] / f"{r['image_id']}.png"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), binary)
            n_done += 1

        return n_done, skipped, time.time() - t_start

    n_written, skipped_no_points, elapsed = _run_inference()
    inference_done = True

    mo.md(
        f"**Inference complete** in {elapsed:.1f}s  \n"
        f"**Predicted masks written**: {n_written}  \n"
        f"**Skipped (mask had < num_points foreground pixels)**: {len(skipped_no_points)}"
    )
    return (inference_done,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute metrics
    """)
    return


@app.cell
def _(calculate_metrics, cv2, inference_done, mo, pairs_df, pl, root):
    mo.stop(not inference_done, mo.md("_(metrics run after inference)_"))

    def _compute_metrics() -> list[dict]:
        rows: list[dict] = []
        for r in pairs_df.iter_rows(named=True):
            pred_path = (
                root / r["dataset"] / r["split"]
                / "PredictedMasks" / r["class_name"] / f"{r['image_id']}.png"
            )
            if not pred_path.is_file():
                continue
            gt_arr = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
            pred_arr = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if gt_arr is None or pred_arr is None:
                continue
            m = calculate_metrics(pred_arr, gt_arr)
            rows.append({
                "dataset": r["dataset"],
                "split": r["split"],
                "class_name": r["class_name"],
                "image_id": r["image_id"],
                **m,
            })
        return rows

    metric_rows = _compute_metrics()

    metrics_df = pl.DataFrame(metric_rows) if metric_rows else pl.DataFrame(
        schema={
            "dataset": pl.String, "split": pl.String, "class_name": pl.String,
            "image_id": pl.String,
            "iou": pl.Float64, "dice": pl.Float64,
            "precision": pl.Float64, "recall": pl.Float64,
        }
    )

    mo.md(f"**Metrics rows**: {metrics_df.height}")
    return (metrics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-class summary
    """)
    return


@app.cell
def _(metrics_df, mo, pl):
    per_class = (
        metrics_df.group_by(["dataset", "class_name"])
        .agg(
            n=pl.len(),
            mean_iou=pl.col("iou").mean(),
            sd_iou=pl.col("iou").std(),
            mean_dice=pl.col("dice").mean(),
            sd_dice=pl.col("dice").std(),
            mean_precision=pl.col("precision").mean(),
            sd_precision=pl.col("precision").std(),
            mean_recall=pl.col("recall").mean(),
            sd_recall=pl.col("recall").std(),
        )
        .sort(["dataset", "class_name"])
    )
    mo.ui.table(per_class, page_size=50)
    return (per_class,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overall summary
    """)
    return


@app.cell
def _(metrics_df, mo, pl):
    overall = (
        metrics_df.group_by("dataset")
        .agg(
            n=pl.len(),
            mean_iou=pl.col("iou").mean(),
            sd_iou=pl.col("iou").std(),
            mean_dice=pl.col("dice").mean(),
            sd_dice=pl.col("dice").std(),
            mean_precision=pl.col("precision").mean(),
            sd_precision=pl.col("precision").std(),
            mean_recall=pl.col("recall").mean(),
            sd_recall=pl.col("recall").std(),
        )
        .sort("dataset")
    )
    mo.ui.table(overall, page_size=10)
    return (overall,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-class Dice (Altair)
    """)
    return


@app.cell
def _(alt, per_class):
    chart = (
        alt.Chart(per_class)
        .mark_bar()
        .encode(
            x=alt.X("class_name:N", sort="-y", title="Class"),
            y=alt.Y("mean_dice:Q", title="Mean Dice"),
            color=alt.Color("dataset:N", title="Dataset"),
            xOffset="dataset:N",
            tooltip=["dataset", "class_name", "n", "mean_dice", "sd_dice"],
        )
        .properties(width=720, height=320)
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save CSVs
    """)
    return


@app.cell
def _(
    Path,
    eval_results_dir,
    metrics_df,
    mo,
    model_preset,
    overall,
    per_class,
    split,
):
    out_dir = Path(eval_results_dir.value).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # File-safe model preset suffix
    suffix = model_preset.value.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")

    paths_written = []
    for name, df in [
        ("per_image", metrics_df),
        ("per_class", per_class),
        ("overall", overall),
    ]:
        p = out_dir / f"m2caiSeg_{split.value}_{suffix}_{name}.csv"
        df.write_csv(p)
        paths_written.append(p)

    mo.md(
        "**Wrote**:\n\n" + "\n".join(f"- `{p}`" for p in paths_written)
    )
    return


if __name__ == "__main__":
    app.run()
