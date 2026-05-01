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
    # Dresden (DSAD) mask preprocessing

    Converts the [Dresden Surgical Anatomy Dataset](https://springernature.figshare.com/articles/dataset/The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_surgical_data_science/21702600)
    into the per-class binary mask layout used elsewhere in this repo.

    ### Source layout

    ```
    {in}/{class_name}/{patient_id:02d}/
        image00.png, image01.png, …   # frames (RGB)
        mask00.png, mask01.png, …     # binary masks (uint8, values {0, 255})
        anno_1/, anno_2/, anno_3/     # alternate annotators — ignored
        weak_labels.csv               # weak labels — ignored
    {in}/multilabel/                  # multi-class masks — ignored
    ```

    The 11 organ classes:
    `abdominal_wall, colon, inferior_mesenteric_artery, intestinal_veins,
    liver, pancreas, small_intestine, spleen, stomach, ureter,
    vesicular_glands`.

    ### Splits — driven by existing CSVs

    The original notebook used unseeded `random.shuffle` per organ class,
    so the split assignment is only preserved in the committed
    `Datasets_AfterPreprocessing/Dresden_{train,val,test}.csv` files.
    This notebook reads those CSVs at startup and uses them as the
    canonical split source — same patient may land in different splits
    for different organs (that's how the original was designed).

    ### Output layout

    ```
    {out}/{train|val|test}/Frames/{class}/{patient}/{frame_id}.png
    {out}/{train|val|test}/Masks/{class}/{patient}/{frame_id}.png
    ```

    `frame_id` is the original numeric stem (e.g. `00`, `01`, ...).

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
    OUTPUT_SPLITS = ("train", "val", "test")
    PATIENT_RE = re.compile(r"^\d+$")
    IMAGE_RE = re.compile(r"^image(\d+)\.png$")
    MASK_RE = re.compile(r"^mask(\d+)\.png$")

    EXPECTED_CLASSES = (
        "abdominal_wall", "colon", "inferior_mesenteric_artery",
        "intestinal_veins", "liver", "pancreas", "small_intestine",
        "spleen", "stomach", "ureter", "vesicular_glands",
    )
    return EXPECTED_CLASSES, IMAGE_RE, MASK_RE, OUTPUT_SPLITS, PATIENT_RE


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell
def _():
    in_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_BeforePreprocessing" / "DSAD"),
        label="Input dataset folder (contains {class}/{patient}/image##.png + mask##.png)",
        full_width=True,
    )
    out_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing" / "Dresden"),
        label="Output dataset folder",
        full_width=True,
    )
    splits_dir = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing"),
        label="Folder containing Dresden_{train,val,test}.csv (canonical splits)",
        full_width=True,
    )
    skip_empty_masks = mo.ui.checkbox(
        value=True,
        label="Skip writing all-zero masks (no segmentation signal)",
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


@app.function
def is_empty_mask(path) -> bool:
    with Image.open(path) as im:
        arr = np.asarray(im)
    return bool(np.all(arr == 0))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load split assignments from CSVs

    Splits come from the previously-committed CSV files
    (`Dresden_{split}.csv`). Each CSV row's `mask` column has the form
    `Dresden/{split}/Masks/{class}/{patient}/{frame}.png` — we extract
    `(class, patient) → split` from those rows.
    """)
    return


@app.cell
def _(splits_dir):
    def load_canonical_splits(splits_root: Path):
        """Return ({class: {patient: split}}, per_split_row_counts)."""
        mapping: dict[str, dict[str, str]] = defaultdict(dict)
        per_split_rows: dict[str, int] = {}
        for split in ("train", "val", "test"):
            csv_path = splits_root / f"Dresden_{split}.csv"
            if not csv_path.is_file():
                per_split_rows[split] = 0
                continue
            df = pl.read_csv(csv_path)
            per_split_rows[split] = df.height
            for mask_rel in df["mask"].to_list():
                parts = mask_rel.split("/")
                # Dresden/<split>/Masks/<class>/<patient>/<frame>.png
                if len(parts) < 6 or parts[2] != "Masks":
                    continue
                cls, patient = parts[3], parts[4]
                prev = mapping[cls].get(patient)
                if prev is not None and prev != split:
                    raise ValueError(
                        f"Inconsistent split for {cls}/{patient}: "
                        f"{prev} vs {split}"
                    )
                mapping[cls][patient] = split
        return dict(mapping), per_split_rows

    splits_root = Path(splits_dir.value).expanduser().resolve()
    _nested, csv_row_counts = (
        load_canonical_splits(splits_root) if splits_root.is_dir() else ({}, {})
    )
    splits_lookup = su.invert_class_video_splits(_nested)

    mo.md(
        f"**Splits CSV root**: `{splits_root}`  \n"
        f"**CSV row counts**: train={csv_row_counts.get('train', 0)}, "
        f"val={csv_row_counts.get('val', 0)}, "
        f"test={csv_row_counts.get('test', 0)}  \n"
        f"**Distinct (class, patient) pairs in CSVs**: {len(splits_lookup)}"
    )
    return (splits_lookup,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discover (class, patient, frame) tuples on disk
    """)
    return


@app.cell
def _(EXPECTED_CLASSES, IMAGE_RE, MASK_RE, PATIENT_RE, in_path, splits_lookup):
    def discover(in_root: Path, splits_map: dict[tuple[str, str], str]):
        records: list[dict] = []
        skipped_no_split: list[tuple[str, str]] = []
        skipped_unknown_class: list[str] = []

        if not in_root.is_dir():
            return records, skipped_no_split, skipped_unknown_class

        for class_dir in sorted(in_root.iterdir()):
            if not class_dir.is_dir():
                continue
            cls = class_dir.name
            if cls == "multilabel":
                continue
            if cls not in EXPECTED_CLASSES:
                skipped_unknown_class.append(cls)
                continue

            for patient_dir in sorted(class_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                patient = patient_dir.name
                if not PATIENT_RE.match(patient):
                    continue

                split = splits_map.get((cls, patient))
                if split is None:
                    skipped_no_split.append((cls, patient))
                    continue

                # Pair image## with mask## by stem id.
                images = {
                    IMAGE_RE.match(f.name).group(1): f
                    for f in patient_dir.iterdir()
                    if f.is_file() and IMAGE_RE.match(f.name)
                }
                masks = {
                    MASK_RE.match(f.name).group(1): f
                    for f in patient_dir.iterdir()
                    if f.is_file() and MASK_RE.match(f.name)
                }
                for frame_id in sorted(images.keys() & masks.keys()):
                    records.append({
                        "class_name": cls,
                        "patient": patient,
                        "frame_id": frame_id,
                        "split": split,
                        "frame_path": str(images[frame_id]),
                        "mask_path": str(masks[frame_id]),
                    })
        return records, skipped_no_split, skipped_unknown_class

    in_root = Path(in_path.value).expanduser().resolve()
    pair_records, skipped_no_split, skipped_unknown = discover(
        in_root, splits_lookup
    )

    if pair_records:
        pairs_df = pl.DataFrame(pair_records)
        per_split_counts = (
            pairs_df.group_by("split").len(name="n_frames").sort("split")
        )
    else:
        pairs_df = pl.DataFrame(
            schema={
                "class_name": pl.String, "patient": pl.String,
                "frame_id": pl.String, "split": pl.String,
                "frame_path": pl.String, "mask_path": pl.String,
            }
        )
        per_split_counts = pl.DataFrame(
            schema={"split": pl.String, "n_frames": pl.UInt32}
        )

    _warn_lines = []
    if skipped_no_split:
        _warn_lines.append(
            f"⚠ {len(skipped_no_split)} (class, patient) folders on disk "
            f"have no split assignment in CSVs (skipped): "
            f"{skipped_no_split[:5]}{'...' if len(skipped_no_split) > 5 else ''}"
        )
    if skipped_unknown:
        _warn_lines.append(
            f"⚠ Unknown class folders skipped: {skipped_unknown}"
        )

    mo.vstack([
        mo.md(
            f"**Input root**: `{in_root}`  \n"
            f"**Frames discovered**: {pairs_df.height}"
            + ("\n\n" + "\n\n".join(_warn_lines) if _warn_lines else "")
        ),
        mo.ui.table(per_split_counts, page_size=10),
    ])
    return (pairs_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pipeline (gated by Run button)
    """)
    return


@app.cell
def _(OUTPUT_SPLITS, out_path, pairs_df, run_button, skip_empty_masks):
    mo.stop(not run_button.value, mo.md("_Press **Run preprocessing** to execute._"))
    mo.stop(pairs_df.height == 0, mo.md("_No frames discovered — check input path._"))

    out_root = Path(out_path.value).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    su.reset_split_dirs(out_root, OUTPUT_SPLITS)

    written_records: list[dict] = []
    skipped_empty: list[dict] = []

    rows = list(pairs_df.iter_rows(named=True))
    for row in mo.status.progress_bar(
        rows, title="Writing Frames + Masks", subtitle="frame/mask pairs",
        completion_title="Pipeline complete",
    ):
        split = row["split"]
        cls = row["class_name"]
        patient = row["patient"]
        frame_id = row["frame_id"]
        mask_src = Path(row["mask_path"])
        frame_src = Path(row["frame_path"])

        if skip_empty_masks.value and is_empty_mask(mask_src):
            skipped_empty.append({
                "split": split, "class": cls, "patient": patient,
                "frame_id": frame_id,
            })
            continue

        binary = su.to_uint8_binary(mask_src)
        mask_dir = out_root / split / "Masks" / cls / patient
        mask_dir.mkdir(parents=True, exist_ok=True)
        with Image.fromarray(binary) as im:
            im.save(mask_dir / f"{frame_id}.png")

        frame_dir = out_root / split / "Frames" / cls / patient
        frame_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame_src, frame_dir / f"{frame_id}.png")

        written_records.append({
            "split": split, "class": cls, "patient": patient,
            "frame_id": frame_id,
        })

    frame_counts = {
        sp: sum(1 for r in written_records if r["split"] == sp)
        for sp in OUTPUT_SPLITS
    }

    mo.md(
        f"**Output root**: `{out_root}`  \n"
        f"**Frame/mask pairs written**: train={frame_counts['train']}, "
        f"val={frame_counts['val']}, test={frame_counts['test']} "
        f"(total {len(written_records)})  \n"
        f"**Empty masks skipped**: {len(skipped_empty)}"
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
        # Dresden uses {class}/{patient}/{frame}.png under both Masks/ and Frames/.
        return frames_root / mask_path.relative_to(masks_root)

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
            f"**Validation passed**: {validation_n} masks all paired "
            f"with same-size frames."
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
