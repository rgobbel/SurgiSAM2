"""Shared helpers for SurgiSAM2 evaluation notebooks.

Each per-dataset eval notebook is a thin shim around these functions. Anything
dataset-specific (frame file extension, dataset name, CSV filename prefix)
is passed in as a parameter — no per-dataset branching lives here.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import altair as alt
import cv2
import numpy as np
import polars as pl
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

def project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until a `pyproject.toml` is found.

    `__file__` is unreliable inside marimo cells (sometimes a bare filename),
    so notebooks should call this with no argument and rely on cwd.
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    for c in candidates:
        for parent in (c, *c.parents):
            if (parent / "pyproject.toml").is_file():
                return parent
    return Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Pure helpers (deterministic, no I/O)
# ---------------------------------------------------------------------------

def select_point_prompt(
    mask: np.ndarray, num_points: int, rng: np.random.Generator,
) -> tuple[list[tuple[int, int]] | None, list[int] | None]:
    """Sample foreground points (mask == 255). Returns (points, labels) or (None, None).

    Points are returned as (x, y) in image coordinates with label 1 (foreground).
    """
    coords = np.argwhere(mask == 255)  # (y, x) pairs
    if len(coords) < num_points:
        return None, None
    idx = rng.choice(len(coords), size=num_points, replace=False)
    pts = [(int(coords[i, 1]), int(coords[i, 0])) for i in idx]
    return pts, [1] * num_points


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """IoU, Dice, Precision, Recall on uint8 masks (>127 = foreground).

    Resizes the prediction to GT size if shapes differ (matches original behavior).
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


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

_PAIR_SCHEMA = {
    "dataset": pl.String,
    "split": pl.String,
    "class_name": pl.String,
    "image_id": pl.String,
    "frame_path": pl.String,
    "mask_path": pl.String,
}


def discover_pairs(
    root: Path, ds_names: list[str], split: str, frame_ext: str = "jpg",
) -> pl.DataFrame:
    """Walk `{root}/{ds}/{split}/Masks/{class}/*.png` and pair with frames.

    `frame_ext` is the extension (without dot) of the corresponding frame
    files in `Frames/`. m2caiSeg/Endoscapes use `jpg`; UD/Dresden/CholecSeg8k
    use `png`. Masks with no matching frame are dropped.
    """
    ext = frame_ext.lstrip(".").lower()
    rows: list[dict] = []
    for ds in ds_names:
        split_dir = root / ds / split
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
                frame = frames_dir / f"{mask.stem}.{ext}"
                if not frame.is_file():
                    continue
                rows.append({
                    "dataset": ds,
                    "split": split,
                    "class_name": class_dir.name,
                    "image_id": mask.stem,
                    "frame_path": str(frame),
                    "mask_path": str(mask),
                })
    if not rows:
        return pl.DataFrame(schema=_PAIR_SCHEMA)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def load_sam2_predictor(
    config_name: str, ckpt_path: Path, sam_configs_dir: Path,
) -> tuple[str, "object"]:
    """Build a `SAM2ImagePredictor` for the given config + checkpoint.

    Resets hydra's global state first so model swaps work mid-session.
    Returns `(device, predictor)` where device is `"cuda"` or `"cpu"`.
    """
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not sam_configs_dir.is_dir():
        raise FileNotFoundError(f"SAM2 config dir not found: {sam_configs_dir}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(sam_configs_dir), version_base=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(
        config_file=config_name, ckpt_path=str(ckpt_path), device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)
    return device, predictor


# ---------------------------------------------------------------------------
# Threaded inference
# ---------------------------------------------------------------------------

def run_threaded_inference(
    *,
    predictor,
    pairs_df: pl.DataFrame,
    root: Path,
    datasets: list[str],
    split: str,
    num_points: int,
    seed: int,
    n_workers: int,
    device: str,
    use_bf16: bool,
    progress_bar_factory: Callable | None = None,
) -> tuple[int, list[str], list[str], float]:
    """Run SAM2 inference on every row of `pairs_df`, writing PredictedMasks/.

    Idempotent: clears `{root}/{ds}/{split}/PredictedMasks/` for each ds first.

    `progress_bar_factory` is a zero-arg callable returning a context manager
    that yields an object with `.update(increment, subtitle)` (i.e. the
    `mo.status.progress_bar` API). Pass `None` for headless / no progress.

    Returns `(n_written, skipped, errors, elapsed_seconds)`.
    """
    bf16_active = bool(use_bf16) and device == "cuda"

    def _autocast():
        if bf16_active:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # Idempotent reset of PredictedMasks for the chosen (datasets, split).
    for ds in datasets:
        pmask_dir = root / ds / split / "PredictedMasks"
        if pmask_dir.exists():
            shutil.rmtree(pmask_dir)

    rows = list(pairs_df.iter_rows(named=True))
    # SeedSequence.spawn gives each work item its own independent substream so
    # sampled prompt points are deterministic across thread counts.
    ss = np.random.SeedSequence(int(seed))
    child_seeds = ss.spawn(len(rows))

    # Predictor mutates shared GPU state inside set_image/predict, so those
    # calls must be serialized. Threads still parallelize image decode + write.
    predictor_lock = threading.Lock()
    output_lock = threading.Lock()
    skipped_lock = threading.Lock()

    skipped: list[str] = []
    errors: list[str] = []
    n_points = int(num_points)

    def _process(idx: int, r: dict) -> bool:
        try:
            with Image.open(r["frame_path"]) as fi:
                frame_rgb = np.array(fi.convert("RGB"))
            with Image.open(r["mask_path"]) as mi:
                gt_mask = np.array(mi.convert("L"))

            rng = np.random.default_rng(child_seeds[idx])
            pts, labels = select_point_prompt(gt_mask, n_points, rng)
            if pts is None:
                with skipped_lock:
                    skipped.append(f"{r['class_name']}/{r['image_id']}")
                return False

            with predictor_lock, _autocast(), torch.no_grad():
                predictor.set_image(frame_rgb)
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
            with output_lock:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), binary)
            return True
        except Exception as e:  # noqa: BLE001 — surface any per-image failure
            with skipped_lock:
                errors.append(f"{r['class_name']}/{r['image_id']}: {e!r}")
            return False

    workers = max(1, int(n_workers))
    t_start = time.time()
    n_done = 0

    if progress_bar_factory is None:
        # Headless path — no progress UI.
        if workers == 1:
            for i, r in enumerate(rows):
                if _process(i, r):
                    n_done += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_process, i, r) for i, r in enumerate(rows)]
                for fut in as_completed(futures):
                    if fut.result():
                        n_done += 1
        return n_done, skipped, errors, time.time() - t_start

    n_processed = 0
    with progress_bar_factory() as bar:
        def _tick(success: bool) -> None:
            nonlocal n_done, n_processed
            if success:
                n_done += 1
            n_processed += 1
            elapsed_so_far = time.time() - t_start
            rate = n_processed / elapsed_so_far if elapsed_so_far > 0 else 0.0
            bar.update(
                increment=1,
                subtitle=(
                    f"{n_done} ok · {len(skipped)} skipped · {len(errors)} err · "
                    f"{rate:.1f} img/s"
                ),
            )

        if workers == 1:
            for i, r in enumerate(rows):
                _tick(_process(i, r))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_process, i, r) for i, r in enumerate(rows)]
                for fut in as_completed(futures):
                    _tick(fut.result())

    return n_done, skipped, errors, time.time() - t_start


# ---------------------------------------------------------------------------
# Metrics computation + summaries
# ---------------------------------------------------------------------------

_METRICS_SCHEMA = {
    "dataset": pl.String, "split": pl.String, "class_name": pl.String,
    "image_id": pl.String,
    "iou": pl.Float64, "dice": pl.Float64,
    "precision": pl.Float64, "recall": pl.Float64,
}


def compute_metrics(pairs_df: pl.DataFrame, root: Path) -> pl.DataFrame:
    """Read GT + predicted masks for every row of `pairs_df` and compute metrics."""
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
    if not rows:
        return pl.DataFrame(schema=_METRICS_SCHEMA)
    return pl.DataFrame(rows)


def summarize_per_class(metrics_df: pl.DataFrame) -> pl.DataFrame:
    return (
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


def summarize_overall(metrics_df: pl.DataFrame) -> pl.DataFrame:
    return (
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


def build_dice_chart(per_class: pl.DataFrame) -> alt.Chart:
    return (
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


def file_safe_suffix(label: str) -> str:
    """Make a model-preset label safe for filenames."""
    return (
        label.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
    )


def save_eval_csvs(
    out_dir: Path,
    *,
    dataset_prefix: str,
    split: str,
    model_label: str,
    metrics_df: pl.DataFrame,
    per_class: pl.DataFrame,
    overall: pl.DataFrame,
) -> list[Path]:
    """Write per_image / per_class / overall CSVs and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = file_safe_suffix(model_label)
    written: list[Path] = []
    for name, df in [
        ("per_image", metrics_df),
        ("per_class", per_class),
        ("overall", overall),
    ]:
        p = out_dir / f"{dataset_prefix}_{split}_{suffix}_{name}.csv"
        df.write_csv(p)
        written.append(p)
    return written
