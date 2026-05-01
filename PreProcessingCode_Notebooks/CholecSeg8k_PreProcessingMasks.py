import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import sys
    import re
    import shutil
    from collections import defaultdict
    from pathlib import Path

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
    # CholecSeg8k mask preprocessing

    Converts the [CholecSeg8k dataset](https://datasetninja.com/cholec-seg8k)
    into the per-class binary mask layout used elsewhere in this repo.

    ### Source layout

    ```
    {in}/{video}/{video}_<offset>/
        frame_<n>_endo.png                # frame
        frame_<n>_endo_color_mask.png     # RGB-coded multi-class mask
        frame_<n>_endo_mask.png           # ignored
        frame_<n>_endo_watershed_mask.png # ignored
    ```

    17 videos × ~5 subfolders × ~80 frames each ≈ 8 080 frames.

    ### Color map (RGB)

    The original notebook used `cv2.imread`, which returns **BGR**, so the
    color tuples there are byte-reversed. We use PIL (RGB) throughout and
    define the colors accordingly.

    | RGB              | Class                  |
    |------------------|------------------------|
    | (210, 140, 140)  | Abdominal Wall         |
    | (255, 0, 0)      | Blood                  |
    | (255, 85, 0)     | Connective Tissue      |
    | (255, 255, 0)    | Cystic Duct            |
    | (186, 183, 75)   | Fat                    |
    | (255, 160, 165)  | Gall bladder           |
    | (231, 70, 156)   | Gastrointestinal Tract |
    | (170, 255, 0)    | Grasper                |
    | (0, 50, 128)     | Hepatic Vein           |
    | (169, 255, 184)  | L-hook Electrocautery  |
    | (255, 114, 114)  | Liver                  |
    | (111, 74, 0)     | Liver Ligament         |

    `(127, 127, 127)` (background) and `(255, 255, 255)` (uncategorized)
    are ignored.

    ### Splits — driven by existing CSVs

    Splits come from `Datasets_AfterPreprocessing/CholecSeg8k_{train,val,test}.csv`.
    The split is per-(class, video): the same video can land in different
    splits for different organ classes. That's how the original was designed.

    **Known issue fixed**: the original CSVs duplicate
    `Abdominal Wall / video27` in both `train` and `test` (a 400-frame
    train↔test leak). We drop those rows from the train side and keep
    them in test.

    ### Output layout

    ```
    {out}/{train|val|test}/Frames/{class}/{video}/{subfolder}/<frame>.png
    {out}/{train|val|test}/Masks/{class}/{video}/{subfolder}/<frame>.png
    ```

    `<frame>` is the raw integer (e.g. `100.png`) extracted from
    `frame_100_endo.png` / `frame_100_endo_color_mask.png`.

    Re-running clears the output split dirs and rewrites — idempotent.
    The source directory is never modified.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constants
    """)
    return


@app.cell
def _():
    # RGB → class name. Derived from the original notebook's BGR map by
    # reversing each tuple.
    CLASS_COLORS_RGB: dict[tuple[int, int, int], str] = {
        (210, 140, 140): "Abdominal Wall",
        (255, 0, 0): "Blood",
        (255, 85, 0): "Connective Tissue",
        (255, 255, 0): "Cystic Duct",
        (186, 183, 75): "Fat",
        (255, 160, 165): "Gall bladder",
        (231, 70, 156): "Gastrointestinal Tract",
        (170, 255, 0): "Grasper",
        (0, 50, 128): "Hepatic Vein",
        (169, 255, 184): "L-hook Electrocautery",
        (255, 114, 114): "Liver",
        (111, 74, 0): "Liver Ligament",
    }
    OUTPUT_SPLITS = ("train", "val", "test")
    EXPECTED_CLASSES = tuple(sorted(CLASS_COLORS_RGB.values()))

    FRAME_RE = re.compile(r"^frame_(\d+)_endo\.png$")
    MASK_RE = re.compile(r"^frame_(\d+)_endo_color_mask\.png$")

    # Original CSVs duplicate this (class, video) pair in both train and test.
    # Per-discussion fix: drop from train, keep in test.
    LEAK_DROP_FROM_TRAIN: tuple[tuple[str, str], ...] = (
        ("Abdominal Wall", "video27"),
    )
    return (
        CLASS_COLORS_RGB,
        FRAME_RE,
        LEAK_DROP_FROM_TRAIN,
        MASK_RE,
        OUTPUT_SPLITS,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell
def _():
    in_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_BeforePreprocessing" / "CholecSeg8k"),
        label="Input dataset folder (contains video##/video##_<offset>/...)",
        full_width=True,
    )
    out_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing" / "CholecSeg8k"),
        label="Output dataset folder",
        full_width=True,
    )
    splits_dir = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing"),
        label="Folder containing CholecSeg8k_{train,val,test}.csv (canonical splits)",
        full_width=True,
    )
    skip_empty_masks = mo.ui.checkbox(
        value=True,
        label="Skip writing all-zero binary masks (no signal for that class)",
    )
    run_button = mo.ui.run_button(label="Run preprocessing")

    mo.vstack([
        in_path, out_path, splits_dir,
        mo.hstack([skip_empty_masks, run_button], justify="start"),
    ])
    return in_path, out_path, run_button, skip_empty_masks, splits_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pure helpers
    """)
    return


@app.cell
def _(CLASS_COLORS_RGB: dict[tuple[int, int, int], str]):
    def split_color_mask(mask_path) -> dict[str, np.ndarray]:
        """Read an RGB color-mask and return {class_name: binary uint8 mask}.

        Only the 12 known organ colors are extracted. (127,127,127) and
        (255,255,255) are ignored. An empty result means no organ class
        was present (or the mask was all background).
        """
        with Image.open(mask_path) as im:
            arr = np.asarray(im.convert("RGB"))

        result: dict[str, np.ndarray] = {}
        for color, class_name in CLASS_COLORS_RGB.items():
            color_arr = np.array(color, dtype=arr.dtype)
            match = np.all(arr == color_arr, axis=-1)
            if not match.any():
                continue
            result[class_name] = (match.astype(np.uint8) * 255)
        return result

    return (split_color_mask,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load split assignments from CSVs

    Each row in `CholecSeg8k_{split}.csv` has a `mask` column of the form
    `CholecSeg8k/{split}/Masks/{class}/{video}/{subfolder}/{frame}.png`.
    We extract `(class, video) → split` from those rows.
    """)
    return


@app.cell
def _(LEAK_DROP_FROM_TRAIN: tuple[tuple[str, str], ...], splits_dir):
    def load_canonical_splits(splits_root: Path):
        mapping: dict[str, dict[str, str]] = defaultdict(dict)
        per_split_rows: dict[str, int] = {}
        conflicts: list[tuple[str, str, str, str]] = []
        for split in ("train", "val", "test"):
            csv_path = splits_root / f"CholecSeg8k_{split}.csv"
            if not csv_path.is_file():
                per_split_rows[split] = 0
                continue
            df = pl.read_csv(csv_path)
            per_split_rows[split] = df.height
            for mask_rel in df["mask"].to_list():
                parts = mask_rel.split("/")
                if "Masks" not in parts:
                    continue
                i = parts.index("Masks")
                if i + 2 >= len(parts):
                    continue
                cls, video = parts[i + 1], parts[i + 2]
                # Apply known-leak fix: drop these from train so the test
                # assignment wins.
                if (cls, video) in LEAK_DROP_FROM_TRAIN and split == "train":
                    continue
                prev = mapping[cls].get(video)
                if prev is not None and prev != split:
                    conflicts.append((cls, video, prev, split))
                mapping[cls][video] = split
        return dict(mapping), per_split_rows, conflicts

    splits_root = Path(splits_dir.value).expanduser().resolve()
    canonical_splits, csv_row_counts, csv_conflicts = (
        load_canonical_splits(splits_root)
        if splits_root.is_dir()
        else ({}, {}, [])
    )

    n_class_video = sum(len(v) for v in canonical_splits.values())
    _conflict_msg = ""
    if csv_conflicts:
        _conflict_msg = (
            "\n\n⚠ Unresolved (class, video) split conflicts: "
            + "; ".join(f"{c}/{v}: {a}↔{b}" for c, v, a, b in csv_conflicts)
        )

    mo.md(
        f"**Splits CSV root**: `{splits_root}`  \n"
        f"**CSV row counts**: train={csv_row_counts.get('train', 0)}, "
        f"val={csv_row_counts.get('val', 0)}, "
        f"test={csv_row_counts.get('test', 0)}  \n"
        f"**Distinct (class, video) pairs after leak-drop**: "
        f"{n_class_video}{_conflict_msg}"
    )
    return (canonical_splits,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discover (video, subfolder, frame) triples on disk
    """)
    return


@app.cell
def _(FRAME_RE, MASK_RE, in_path):
    def discover(in_root: Path) -> list[dict]:
        """Walk in_root and return one record per (frame, mask) pair found."""
        records: list[dict] = []
        if not in_root.is_dir():
            return records

        for video_dir in sorted(in_root.iterdir()):
            if not video_dir.is_dir() or not video_dir.name.startswith("video"):
                continue
            video = video_dir.name
            for sub_dir in sorted(video_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                subfolder = sub_dir.name

                frames: dict[str, Path] = {}
                masks: dict[str, Path] = {}
                for f in sub_dir.iterdir():
                    if not f.is_file():
                        continue
                    fm = FRAME_RE.match(f.name)
                    if fm:
                        frames[fm.group(1)] = f
                        continue
                    mm = MASK_RE.match(f.name)
                    if mm:
                        masks[mm.group(1)] = f

                for frame_id in sorted(frames.keys() & masks.keys()):
                    records.append({
                        "video": video,
                        "subfolder": subfolder,
                        "frame_id": frame_id,
                        "frame_path": str(frames[frame_id]),
                        "mask_path": str(masks[frame_id]),
                    })
        return records

    in_root = Path(in_path.value).expanduser().resolve()
    pair_records = discover(in_root) if in_root.is_dir() else []

    if pair_records:
        pairs_df = pl.DataFrame(pair_records)
        per_video_counts = (
            pairs_df.group_by("video").len(name="n_frames").sort("video")
        )
    else:
        pairs_df = pl.DataFrame(
            schema={
                "video": pl.String, "subfolder": pl.String,
                "frame_id": pl.String, "frame_path": pl.String,
                "mask_path": pl.String,
            }
        )
        per_video_counts = pl.DataFrame(
            schema={"video": pl.String, "n_frames": pl.UInt32}
        )

    mo.vstack([
        mo.md(
            f"**Input root**: `{in_root}`  \n"
            f"**Frame/mask pairs discovered**: {pairs_df.height}"
        ),
        mo.ui.table(per_video_counts, page_size=20),
    ])
    return (pairs_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pipeline (gated by Run button)
    """)
    return


@app.cell
def _(
    OUTPUT_SPLITS,
    canonical_splits,
    out_path,
    pairs_df,
    run_button,
    skip_empty_masks,
    split_color_mask,
):
    mo.stop(not run_button.value, mo.md("_Press **Run preprocessing** to execute._"))
    mo.stop(pairs_df.height == 0, mo.md("_No frames discovered — check input path._"))
    mo.stop(
        not canonical_splits,
        mo.md("_Canonical splits empty — check splits CSV folder._"),
    )

    out_root = Path(out_path.value).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    su.reset_split_dirs(out_root, OUTPUT_SPLITS)

    cv_to_split = su.invert_class_video_splits(canonical_splits)

    written_records: list[dict] = []
    skipped_no_assignment: list[dict] = []
    skipped_empty: list[dict] = []
    copied_frames: dict[tuple[str, str, str, str], None] = {}

    n_total = pairs_df.height
    for row in mo.status.progress_bar(
        pairs_df.iter_rows(named=True),
        title="Splitting masks",
        total=n_total,
    ):
        video = row["video"]
        subfolder = row["subfolder"]
        frame_id = row["frame_id"]
        mask_src = Path(row["mask_path"])
        frame_src = Path(row["frame_path"])

        class_masks = split_color_mask(mask_src)
        if not class_masks:
            skipped_empty.append({
                "video": video, "subfolder": subfolder, "frame_id": frame_id,
            })
            continue

        for class_name, binary in class_masks.items():
            split = cv_to_split.get((class_name, video))
            if split is None:
                skipped_no_assignment.append({
                    "video": video, "subfolder": subfolder,
                    "frame_id": frame_id, "class": class_name,
                })
                continue
            if skip_empty_masks.value and not binary.any():
                continue

            mask_dir = (
                out_root / split / "Masks" / class_name / video / subfolder
            )
            mask_dir.mkdir(parents=True, exist_ok=True)
            with Image.fromarray(binary) as im:
                im.save(mask_dir / f"{frame_id}.png")

            key = (split, video, subfolder, frame_id)
            if key not in copied_frames:
                frame_dir = (
                    out_root / split / "Frames" / video / subfolder
                )
                frame_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frame_src, frame_dir / f"{frame_id}.png")
                copied_frames[key] = None

            written_records.append({
                "split": split, "class": class_name, "video": video,
                "subfolder": subfolder, "frame_id": frame_id,
            })

    mask_counts = {
        sp: sum(1 for r in written_records if r["split"] == sp)
        for sp in OUTPUT_SPLITS
    }
    frame_counts = {
        sp: sum(1 for k in copied_frames if k[0] == sp)
        for sp in OUTPUT_SPLITS
    }

    _no_assign_msg = (
        f"\n  *(skipped — class present on disk but no split assignment for "
        f"that (class, video) pair: {len(skipped_no_assignment)})*"
        if skipped_no_assignment else ""
    )

    mo.md(
        f"**Output root**: `{out_root}`  \n"
        f"**Frames written**: train={frame_counts['train']}, "
        f"val={frame_counts['val']}, test={frame_counts['test']}  \n"
        f"**Class-mask files written**: train={mask_counts['train']}, "
        f"val={mask_counts['val']}, test={mask_counts['test']} "
        f"(total {len(written_records)})  \n"
        f"**Frames with no organ class on disk**: {len(skipped_empty)}"
        f"{_no_assign_msg}"
    )
    return out_root, written_records


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation — every mask has a frame and shapes match
    """)
    return


@app.cell
def _(OUTPUT_SPLITS, out_root, run_button):
    mo.stop(not run_button.value, mo.md("_(validation runs after pipeline)_"))

    def _frame_for_mask(mask_path, masks_root, frames_root):
        # Mask layout: Masks/{class}/{video}/{subfolder}/<frame>.png
        # Frame layout: Frames/{video}/{subfolder}/<frame>.png — strip leading class.
        rel = mask_path.relative_to(masks_root)
        return frames_root / Path(*rel.parts[1:])

    validation_issues, validation_n = su.validate_frames_masks_paired(
        out_root, OUTPUT_SPLITS, frame_for_mask=_frame_for_mask,
    )

    if validation_issues:
        body = "\n".join(f"- {line}" for line in validation_issues[:50])
        more = (
            "" if len(validation_issues) <= 50
            else f"\n…and {len(validation_issues) - 50} more"
        )
        mo.md(
            f"**Validation: {len(validation_issues)} issue(s)** "
            f"out of {validation_n} masks\n\n{body}{more}"
        )
    else:
        mo.md(
            f"**Validation passed**: {validation_n} masks all paired with "
            f"same-size frames."
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
