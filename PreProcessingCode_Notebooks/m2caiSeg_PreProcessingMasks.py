import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import sys
    import shutil
    from collections import Counter
    from pathlib import Path

    import cv2
    import marimo as mo
    import numpy as np
    import polars as pl
    from PIL import Image

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
    # m2caiSeg mask preprocessing

    Converts the raw m2caiSeg dataset (RGB-coded masks under
    `train/`, `test/`, `trainval/`) into a clean, split-by-image dataset
    with one binary PNG per organ class per image.

    ### Splits — driven by existing CSVs

    The original notebook used unseeded `random.shuffle`, so the only
    canonical record of the splits is in
    `Datasets_AfterPreprocessing/m2caiSeg_{train,val,test}.csv`. This
    notebook reads those CSVs at startup and uses them as the source of
    truth (one split per `image_id`). A fresh shuffle would produce
    different splits and invalidate previously cached eval results.

    ### Output layout

    ```
    {out}/{train|val|test}/Frames/{image_id}.jpg
    {out}/{train|val|test}/Masks/{class_name}/{image_id}.png
    ```

    Re-running clears the output split dirs and rewrites — idempotent.
    The source directory is never modified.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constants — class colors
    """)
    return


@app.cell
def _():
    # RGB color → snake_case class name. Hyphens replaced with underscores per
    # CLAUDE.md (gall-bladder → gall_bladder, specimen-bag → specimen_bag).
    # gall_bladder (85, 170, 255) is intentionally kept (was dropped in the
    # original notebook's allowed_colors list).
    CLASS_COLORS: dict[tuple[int, int, int], str] = {
        (170, 0, 85): "unknown",
        (0, 85, 170): "grasper",
        (0, 85, 255): "bipolar",
        (0, 170, 85): "hook",
        (0, 255, 85): "scissors",
        (0, 255, 170): "clipper",
        (85, 0, 170): "irrigator",
        (85, 0, 255): "specimen_bag",
        (170, 85, 85): "trocars",
        (170, 170, 170): "clip",
        (85, 170, 0): "liver",
        (85, 170, 255): "gall_bladder",
        (85, 255, 0): "fat",
        (85, 255, 170): "upperwall",
        (170, 0, 255): "artery",
        (255, 0, 255): "intestine",
        (255, 255, 0): "bile",
        (255, 0, 0): "blood",
    }

    # Black (0, 0, 0) is treated as background and skipped — it appeared in the
    # original allowed_colors list but it is not an organ class.

    def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "".join(f"{c:02x}" for c in rgb)

    return (CLASS_COLORS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell
def _():
    in_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_BeforePreprocessing" / "m2caiSeg"),
        label="Input dataset folder",
        full_width=True,
    )
    out_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing" / "m2caiSeg"),
        label="Output dataset folder",
        full_width=True,
    )
    splits_dir = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing"),
        label="Folder containing m2caiSeg_{train,val,test}.csv (canonical splits)",
        full_width=True,
    )
    min_color_pixels = mo.ui.number(
        start=0, stop=10000, step=1, value=75,
        label="Min pixel area for a color to be kept in a mask",
    )
    min_component_pixels = mo.ui.number(
        start=0, stop=10000, step=1, value=50,
        label="Min largest-component area for a per-class binary mask",
    )
    run_button = mo.ui.run_button(label="Run preprocessing")

    mo.vstack([
        in_path, out_path, splits_dir,
        mo.hstack([min_color_pixels, min_component_pixels], justify="start"),
        run_button,
    ])
    return (
        in_path,
        min_color_pixels,
        min_component_pixels,
        out_path,
        run_button,
        splits_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pure helpers
    """)
    return


@app.cell
def _():
    def get_color_counts(mask_path) -> Counter:
        with Image.open(mask_path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb).reshape(-1, 3)
        # Counter over RGB tuples
        return Counter(map(tuple, arr.tolist()))

    def filter_small_colors(
        counts: Counter, min_pixels: int
    ) -> set[tuple[int, int, int]]:
        """Return colors whose total area meets `min_pixels`."""
        return {color for color, n in counts.items() if n >= min_pixels}

    def extract_color_mask(mask_path, color: tuple[int, int, int]) -> np.ndarray:
        """Binary uint8 mask (0 / 255) for pixels equal to `color`."""
        with Image.open(mask_path) as im:
            arr = np.asarray(im.convert("RGB"))
        match = np.all(arr == np.array(color, dtype=arr.dtype), axis=-1)
        return (match.astype(np.uint8) * 255)

    def largest_component_area(binary: np.ndarray) -> int:
        """Largest non-background connected-component area in a binary mask."""
        if binary.max() == 0:
            return 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return 0
        return int(max(stats[i, cv2.CC_STAT_AREA] for i in range(1, n)))

    return (
        extract_color_mask,
        filter_small_colors,
        get_color_counts,
        largest_component_area,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discover input frames + masks

    m2caiSeg ships three subfolders (`train`, `test`, `trainval`). We pool
    all `(image_id, frame_path, mask_path)` tuples and dedupe by `image_id`
    — the train/val/test split is reassigned downstream by image, so the
    original folder split is informational only.
    """)
    return


@app.cell
def _(in_path):
    INPUT_SUBSPLITS = ("train", "test", "trainval")

    def discover_pairs(in_root: Path) -> dict[str, dict]:
        """image_id → {frame, mask, source_split}. Dedupe by image_id."""
        pairs: dict[str, dict] = {}
        for sub in INPUT_SUBSPLITS:
            img_dir = in_root / sub / "images"
            mask_dir = in_root / sub / "groundtruth"
            if not img_dir.is_dir() or not mask_dir.is_dir():
                continue
            for frame in sorted(img_dir.iterdir()):
                if frame.suffix.lower() != ".jpg":
                    continue
                image_id = frame.stem
                mask = mask_dir / f"{image_id}_gt.png"
                if not mask.is_file():
                    continue
                pairs.setdefault(
                    image_id,
                    {"frame": frame, "mask": mask, "source_split": sub},
                )
        return pairs

    in_root = Path(in_path.value).expanduser().resolve()
    discovered = discover_pairs(in_root) if in_root.is_dir() else {}

    mo.md(
        f"**Input root**: `{in_root}`  \n"
        f"**Unique image_ids discovered**: {len(discovered)}"
    )
    return (discovered,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load split assignments from CSVs

    Each row in `m2caiSeg_{split}.csv` has a `mask` column of the form
    `m2caiSeg/{split}/Masks/{class_name}/{image_id}.png`. We extract
    `image_id → split` from those rows.
    """)
    return


@app.cell
def _(splits_dir):
    def load_canonical_splits(splits_root: Path):
        image_to_split: dict[str, str] = {}
        per_split_rows: dict[str, int] = {}
        conflicts: list[tuple[str, str, str]] = []
        for split in ("train", "val", "test"):
            csv_path = splits_root / f"m2caiSeg_{split}.csv"
            if not csv_path.is_file():
                per_split_rows[split] = 0
                continue
            df = pl.read_csv(csv_path)
            per_split_rows[split] = df.height
            for mask_rel in df["mask"].to_list():
                image_id = Path(mask_rel).stem
                prev = image_to_split.get(image_id)
                if prev is not None and prev != split:
                    conflicts.append((image_id, prev, split))
                    continue
                image_to_split[image_id] = split
        return image_to_split, per_split_rows, conflicts

    splits_root = Path(splits_dir.value).expanduser().resolve()
    image_to_split, csv_row_counts, csv_conflicts = (
        load_canonical_splits(splits_root)
        if splits_root.is_dir()
        else ({}, {}, [])
    )

    _conflict_msg = ""
    if csv_conflicts:
        _conflict_msg = (
            "\n\n⚠ Inconsistent split for image_ids: "
            + "; ".join(f"{i}: {a}↔{b}" for i, a, b in csv_conflicts[:10])
            + ("…" if len(csv_conflicts) > 10 else "")
        )

    mo.md(
        f"**Splits CSV root**: `{splits_root}`  \n"
        f"**CSV row counts**: train={csv_row_counts.get('train', 0)}, "
        f"val={csv_row_counts.get('val', 0)}, "
        f"test={csv_row_counts.get('test', 0)}  \n"
        f"**Distinct image_ids in CSVs**: {len(image_to_split)}{_conflict_msg}"
    )
    return (image_to_split,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pipeline (gated by Run button)
    """)
    return


@app.cell
def _(
    CLASS_COLORS: dict[tuple[int, int, int], str],
    discovered,
    extract_color_mask,
    filter_small_colors,
    get_color_counts,
    image_to_split,
    largest_component_area,
    min_color_pixels,
    min_component_pixels,
    out_path,
    run_button,
):
    mo.stop(not run_button.value, mo.md("_Press **Run preprocessing** to execute._"))
    mo.stop(not discovered, mo.md("_No input pairs discovered — check input path._"))
    mo.stop(
        not image_to_split,
        mo.md("_Canonical splits empty — check splits CSV folder._"),
    )

    OUTPUT_SPLITS = ("train", "val", "test")

    def write_pair(
        image_id: str,
        frame_path: Path,
        mask_path: Path,
        split_dir: Path,
        min_color: int,
        min_cc: int,
    ) -> dict[str, int]:
        """Process one image. Returns per-class counts written ({} if frame skipped)."""
        counts = get_color_counts(mask_path)
        kept_colors = filter_small_colors(counts, min_color)

        # Per-class binary masks (only for allowed classes whose color survived).
        per_class: dict[str, int] = {}
        for color, class_name in CLASS_COLORS.items():
            if color not in kept_colors:
                continue
            binary = extract_color_mask(mask_path, color)
            if largest_component_area(binary) < min_cc:
                continue
            class_dir = split_dir / "Masks" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            out = class_dir / f"{image_id}.png"
            with Image.fromarray(binary) as im:
                im.save(out)
            per_class[class_name] = 1

        if not per_class:
            return {}

        # Copy frame only when at least one class mask was written.
        frames_dir = split_dir / "Frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame_path, frames_dir / f"{image_id}.jpg")
        return per_class

    out_root = Path(out_path.value).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    su.reset_split_dirs(out_root, OUTPUT_SPLITS)

    # Pair every discovered image_id with its canonical split. image_ids that
    # appear on disk but not in the CSVs are skipped; the reverse case
    # (CSV-only) is surfaced separately.
    assigned: list[tuple[str, str]] = []  # (split, image_id)
    skipped_no_assignment: list[str] = []
    for image_id in sorted(discovered.keys()):
        sp = image_to_split.get(image_id)
        if sp is None:
            skipped_no_assignment.append(image_id)
            continue
        assigned.append((sp, image_id))
    missing_on_disk = sorted(set(image_to_split) - set(discovered))

    written_records: list[dict] = []
    skipped_no_class: list[str] = []
    for split, image_id in mo.status.progress_bar(
        assigned, title="Writing Frames + Masks", subtitle="images",
        completion_title="Pipeline complete",
    ):
        split_dir = out_root / split
        entry = discovered[image_id]
        classes = write_pair(
            image_id,
            entry["frame"],
            entry["mask"],
            split_dir,
            min_color_pixels.value,
            min_component_pixels.value,
        )
        if not classes:
            skipped_no_class.append(image_id)
            continue
        for class_name in classes:
            written_records.append(
                {"split": split, "class": class_name, "image_id": image_id}
            )

    placed_counts = {
        sp: sum(1 for s, _ in assigned if s == sp) for sp in OUTPUT_SPLITS
    }

    mo.md(
        f"**Output root**: `{out_root}`  \n"
        f"**Images placed (canonical)**: train={placed_counts['train']}, "
        f"val={placed_counts['val']}, test={placed_counts['test']}  \n"
        f"**Frames written** (≥1 class survived filters): "
        f"{len({(r['split'], r['image_id']) for r in written_records})}  \n"
        f"**Frames skipped** (no class survived filters): "
        f"{len(skipped_no_class)}  \n"
        f"**Class-mask files written**: {len(written_records)}  \n"
        f"**On disk but not in CSVs (skipped)**: {len(skipped_no_assignment)}  \n"
        f"**In CSVs but missing on disk**: {len(missing_on_disk)}"
    )
    return OUTPUT_SPLITS, out_root, written_records


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation — every mask has a frame and shapes match
    """)
    return


@app.cell
def _(OUTPUT_SPLITS, out_root, run_button):
    mo.stop(not run_button.value, mo.md("_(validation runs after pipeline)_"))

    validation_issues, validation_n = su.validate_frames_masks_paired(
        out_root, OUTPUT_SPLITS, frame_ext="jpg",
    )

    if validation_issues:
        body = "\n".join(f"- {line}" for line in validation_issues[:50])
        more = "" if len(validation_issues) <= 50 else f"\n…and {len(validation_issues) - 50} more"
        mo.md(
            f"**Validation: {len(validation_issues)} issue(s)** "
            f"out of {validation_n} masks\n\n{body}{more}"
        )
    else:
        mo.md(
            f"**Validation passed**: {validation_n} class-masks all paired with same-size frames."
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Per-class counts per split
    """)
    return


@app.cell
def _(run_button, written_records: list[dict]):
    mo.stop(not run_button.value, mo.md("_(table populated after pipeline)_"))

    counts_df = su.build_counts_df(written_records)
    mo.ui.table(counts_df, page_size=50)
    return (counts_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Class counts per split (Altair)
    """)
    return


@app.cell
def _(counts_df):
    chart = su.class_counts_chart(counts_df)
    chart
    return


if __name__ == "__main__":
    app.run()
