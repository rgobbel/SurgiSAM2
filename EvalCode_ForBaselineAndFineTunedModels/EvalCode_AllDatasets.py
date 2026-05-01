import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    # Walk up from cwd until pyproject.toml is found; ensure that root is on
    # sys.path so `import surgisam_utils` works regardless of where marimo
    # was launched from. `__file__` is unreliable inside marimo cells.
    def _find_project_root() -> Path:
        for parent in (Path.cwd(), *Path.cwd().parents):
            if (parent / "pyproject.toml").is_file():
                return parent
        return Path.cwd()

    PROJECT_ROOT = _find_project_root()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import surgisam_utils as su


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Multi-dataset evaluation — SAM2 baseline / fine-tuned

    Runs SAM2 inference on any subset of the five preprocessed datasets
    (`m2caiSeg`, `Endoscapes`, `UD`, `Dresden`, `CholecSeg8k`) and computes
    IoU / Dice / Precision / Recall per class.

    Source of pairs: the canonical
    `Datasets_AfterPreprocessing/{Dataset}_{split}.csv` files. The CSVs
    encode each frame ↔ mask pair explicitly, which means this notebook
    works for both flat layouts (m2caiSeg, Endoscapes, UD) and nested
    layouts (Dresden, CholecSeg8k) without any layout-aware code.

    Predictions land at the same path as the GT mask but with `Masks/`
    replaced by `PredictedMasks/` — preserving any nesting:

    ```
    {root}/{ds}/{split}/PredictedMasks/{class}/[nested...]/{stem}.png
    ```

    Re-running clears `PredictedMasks/` for each chosen `(dataset, split)`
    before writing — idempotent at the predicted-masks tier.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell(hide_code=True)
def _():
    DATASET_OPTIONS = ("m2caiSeg", "Endoscapes", "UD", "Dresden", "CholecSeg8k")

    dataset_root = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing"),
        label="Dataset root (also holds {Dataset}_{split}.csv split files)",
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
        options=list(DATASET_OPTIONS),
        value=list(DATASET_OPTIONS),
        label="Datasets",
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
            str(PROJECT_ROOT / "checkpoints" / "Curated400_checkpoint_26.pt"),
        ),
    }
    model_preset = mo.ui.dropdown(
        options=list(MODEL_PRESETS.keys()),
        value="Hiera base+ (baseline)",
        label="Model preset",
    )

    num_points = mo.ui.number(start=1, stop=50, step=1, value=1, label="Points per prompt")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=42, label="Random seed")
    max_workers = mo.ui.number(
        start=1, stop=64, step=1, value=2,
        label="Worker threads (1 = serial)",
    )
    use_bf16 = mo.ui.checkbox(
        value=True,
        label="bf16 autocast (CUDA only)",
    )

    load_button = mo.ui.run_button(label="Load model")
    run_button = mo.ui.run_button(label="Run inference + metrics")
    return (
        MODEL_PRESETS,
        dataset_root,
        datasets,
        eval_results_dir,
        load_button,
        max_workers,
        model_preset,
        num_points,
        run_button,
        sam_configs_dir,
        seed,
        split,
        use_bf16,
    )


@app.cell(hide_code=True)
def _(
    dataset_root,
    datasets,
    eval_results_dir,
    load_button,
    max_workers,
    model_preset,
    num_points,
    run_button,
    sam_configs_dir,
    seed,
    split,
    use_bf16,
):
    mo.vstack([
        dataset_root,
        sam_configs_dir,
        eval_results_dir,
        mo.hstack([datasets, split], justify="start"),
        mo.hstack([model_preset, num_points, seed, max_workers, use_bf16], justify="start"),
        mo.hstack([load_button, run_button], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discover (frame, mask) pairs

    Reads `{dataset_root}/{ds}_{split}.csv` for each selected dataset.
    Path values in the CSVs are relative to `dataset_root`.
    """)
    return


@app.cell(hide_code=True)
def _(dataset_root, datasets, split):
    root = Path(dataset_root.value).expanduser().resolve()
    pairs_df = su.load_pairs_from_csvs(root, list(datasets.value), split.value)

    if pairs_df.height:
        summary = (
            pairs_df.group_by(["dataset", "class_name"])
            .len(name="n_pairs")
            .sort(["dataset", "class_name"])
        )
        per_dataset = (
            pairs_df.group_by("dataset").len(name="n_pairs").sort("dataset")
        )
    else:
        summary = pl.DataFrame(
            schema={"dataset": pl.String, "class_name": pl.String, "n_pairs": pl.UInt32}
        )
        per_dataset = pl.DataFrame(
            schema={"dataset": pl.String, "n_pairs": pl.UInt32}
        )

    mo.vstack([
        mo.md(
            f"**Root**: `{root}`  \n"
            f"**Pairs discovered**: {pairs_df.height}"
        ),
        mo.ui.table(per_dataset, page_size=10),
        mo.ui.table(summary, page_size=50),
    ])
    return pairs_df, root


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load model

    Press **Load model** to (re)build the SAM2 predictor with the selected
    preset. The cell re-runs whenever the preset changes and the load
    button is still latched on.
    """)
    return


@app.cell(hide_code=True)
def _(MODEL_PRESETS, load_button, model_preset, sam_configs_dir):
    mo.stop(not load_button.value, mo.md("_Press **Load model** to build the predictor._"))

    config_name, ckpt_rel = MODEL_PRESETS[model_preset.value]
    ckpt_path = Path(ckpt_rel).expanduser().resolve()
    cfg_dir = Path(sam_configs_dir.value).expanduser().resolve()

    try:
        device, predictor = su.load_sam2_predictor(config_name, ckpt_path, cfg_dir)
    except FileNotFoundError as e:
        mo.stop(True, mo.md(f"❌ {e}"))

    mo.md(
        f"**Model loaded**: {model_preset.value}  \n"
        f"**Device**: `{device}`  \n"
        f"**Checkpoint**: `{ckpt_path}`  \n"
        f"**Config**: `{cfg_dir}/{config_name}`"
    )
    return device, predictor


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run inference

    For each (frame, mask) pair, samples `num_points` foreground points
    from the GT mask, calls SAM2, and writes a binary predicted mask.
    The predicted-masks tree for each chosen `(dataset, split)` is
    cleared before the run.
    """)
    return


@app.cell(hide_code=True)
def _(
    datasets,
    device,
    max_workers,
    num_points,
    pairs_df,
    predictor,
    root,
    run_button,
    seed,
    split,
    use_bf16,
):
    mo.stop(not run_button.value, mo.md("_Press **Run inference + metrics** to start._"))
    mo.stop(pairs_df.height == 0, mo.md("_No pairs discovered — check dataset root._"))

    n_workers = max(1, int(max_workers.value))
    bf16_active = bool(use_bf16.value) and device == "cuda"

    def _bar_factory():
        return mo.status.progress_bar(
            total=pairs_df.height,
            title=f"SAM2 inference ({n_workers} worker{'s' if n_workers != 1 else ''})",
            subtitle="starting…",
            remove_on_exit=False,
        )

    n_written, skipped_no_points, error_list, elapsed = su.run_threaded_inference(
        predictor=predictor,
        pairs_df=pairs_df,
        root=root,
        datasets=list(datasets.value),
        split=split.value,
        num_points=num_points.value,
        seed=seed.value,
        n_workers=n_workers,
        device=device,
        use_bf16=use_bf16.value,
        progress_bar_factory=_bar_factory,
    )
    inference_done = True

    _err_md = ""
    if error_list:
        _shown = "\n".join(f"- `{e}`" for e in error_list[:10])
        _more = "" if len(error_list) <= 10 else f"\n- _…and {len(error_list) - 10} more_"
        _err_md = f"\n\n**Errors**: {len(error_list)}\n{_shown}{_more}"

    _precision_label = "bf16 autocast" if bf16_active else (
        "fp32 (bf16 requested but device is CPU)" if use_bf16.value else "fp32"
    )

    mo.md(
        f"**Inference complete** in {elapsed:.1f}s  \n"
        f"**Workers**: {n_workers}  \n"
        f"**Precision**: {_precision_label}  \n"
        f"**Predicted masks written**: {n_written}  \n"
        f"**Skipped (mask had < num_points foreground pixels)**: {len(skipped_no_points)}"
        f"{_err_md}"
    )
    return (inference_done,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compute metrics
    """)
    return


@app.cell(hide_code=True)
def _(inference_done, pairs_df, root):
    mo.stop(not inference_done, mo.md("_(metrics run after inference)_"))

    metrics_df = su.compute_metrics(pairs_df, root)
    mo.md(
        f"**Metrics rows**: {metrics_df.height} "
        f"(of {pairs_df.height} pairs — gap = predictions missing on disk)"
    )
    return (metrics_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Per-class summary
    """)
    return


@app.cell(hide_code=True)
def _(metrics_df):
    per_class = su.summarize_per_class(metrics_df)
    mo.ui.table(per_class, page_size=80)
    return (per_class,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Overall summary (per dataset)
    """)
    return


@app.cell(hide_code=True)
def _(metrics_df):
    overall = su.summarize_overall(metrics_df)
    mo.ui.table(overall, page_size=20)
    return (overall,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Per-class Dice (Altair)
    """)
    return


@app.cell(hide_code=True)
def _(per_class):
    chart = su.build_dice_chart(per_class)
    chart
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save CSVs

    Writes one bundle of CSVs (`per_image`, `per_class`, `overall`) per
    selected dataset to `eval_results_dir`.
    """)
    return


@app.cell(hide_code=True)
def _(eval_results_dir, metrics_df, model_preset, overall, per_class, split):
    out_dir = Path(eval_results_dir.value).expanduser().resolve()
    paths_written: list[Path] = []
    for ds in sorted(metrics_df["dataset"].unique().to_list()):
        ds_metrics = metrics_df.filter(pl.col("dataset") == ds)
        ds_per_class = per_class.filter(pl.col("dataset") == ds)
        ds_overall = overall.filter(pl.col("dataset") == ds)
        if ds_metrics.height == 0:
            continue
        paths_written.extend(su.save_eval_csvs(
            out_dir,
            dataset_prefix=ds,
            split=split.value,
            model_label=model_preset.value,
            metrics_df=ds_metrics,
            per_class=ds_per_class,
            overall=ds_overall,
        ))

    mo.md(
        "**Wrote**:\n\n" + "\n".join(f"- `{p}`" for p in paths_written)
        if paths_written
        else "_(nothing to write — metrics were empty)_"
    )
    return


if __name__ == "__main__":
    app.run()
