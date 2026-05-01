import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import sys
    import shutil
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
    # Endoscapes mask preprocessing

    Converts the raw [Endoscapes2023 dataset](https://github.com/CAMMA-public/Endoscapes)
    instance-segmentation arrays into the per-class binary mask layout used
    elsewhere in this repo.

    ### Source layout

    ```
    {in}/
        insseg/{stem}.npy   # (N, H, W) uint8 masks, values in {0, 1}
        insseg/{stem}.csv   # N rows, each row is a float class id (1–6)
        train_seg/{stem}.jpg
        val_seg/{stem}.jpg
        test_seg/{stem}.jpg
    ```

    The 493 `insseg` stems partition exactly into `train_seg / val_seg /
    test_seg` (343 / 76 / 74). Split assignment is taken directly from
    that folder membership — no shuffle, no seed.

    ### Class id → name (from `seg_label_map.txt`)

    | id | name                  |
    |----|-----------------------|
    | 1  | Cystic Plate          |
    | 2  | Hepatocystic triangle |
    | 3  | Cystic Artery         |
    | 4  | Cystic Duct           |
    | 5  | Gall Bladder          |
    | 6  | Instruments           |

    Multiple class-6 instances per frame are bitwise-OR'd into a single
    `Instruments` mask. Mixed-case spaced names match the existing
    `Endoscapes_{split}.csv` files in `Datasets_AfterPreprocessing/`.

    ### Output layout

    ```
    {out}/{train|val|test}/Frames/{stem}.jpg
    {out}/{train|val|test}/Masks/{class_name}/{stem}.png
    ```

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


@app.cell(hide_code=True)
def _():
    CLASS_ID_TO_NAME: dict[int, str] = {
        1: "Cystic Plate",
        2: "Hepatocystic triangle",
        3: "Cystic Artery",
        4: "Cystic Duct",
        5: "Gall Bladder",
        6: "Instruments",
    }
    OUTPUT_SPLITS = ("train", "val", "test")
    SEG_FOLDER_FOR_SPLIT = {
        "train": "train_seg",
        "val": "val_seg",
        "test": "test_seg",
    }
    return CLASS_ID_TO_NAME, OUTPUT_SPLITS, SEG_FOLDER_FOR_SPLIT


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell(hide_code=True)
def _():
    in_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_BeforePreprocessing" / "endoscapes" / "endoscapes"),
        label="Input dataset folder (contains insseg/ and *_seg/)",
        full_width=True,
    )
    out_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing" / "Endoscapes"),
        label="Output dataset folder",
        full_width=True,
    )
    skip_empty_masks = mo.ui.checkbox(
        value=True,
        label="Skip writing all-zero masks (no segmentation signal)",
    )
    run_button = mo.ui.run_button(label="Run preprocessing")

    mo.vstack([
        in_path, out_path,
        mo.hstack([skip_empty_masks, run_button], justify="start"),
    ])
    return in_path, out_path, run_button, skip_empty_masks


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pure helpers
    """)
    return


@app.cell(hide_code=True)
def _():
    def is_empty(arr: np.ndarray) -> bool:
        return bool(np.all(arr == 0))

    def consolidate(arrs: list[np.ndarray]) -> np.ndarray:
        """Bitwise-OR a list of binary uint8 masks (all same shape)."""
        out = arrs[0].copy()
        for a in arrs[1:]:
            np.bitwise_or(out, a, out=out)
        return out

    return consolidate, is_empty


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discover (npy, csv, frame, split) tuples
    """)
    return


@app.cell(hide_code=True)
def _(SEG_FOLDER_FOR_SPLIT, in_path):
    def discover(in_root: Path) -> list[dict]:
        insseg = in_root / "insseg"
        records: list[dict] = []
        if not insseg.is_dir():
            return records

        seg_dirs = {
            split: in_root / sub for split, sub in SEG_FOLDER_FOR_SPLIT.items()
        }
        seg_stems = {
            split: (
                {f.stem for f in d.iterdir() if f.suffix.lower() == ".jpg"}
                if d.is_dir()
                else set()
            )
            for split, d in seg_dirs.items()
        }

        for npy_file in sorted(insseg.iterdir()):
            if npy_file.suffix.lower() != ".npy":
                continue
            stem = npy_file.stem
            csv_file = insseg / f"{stem}.csv"
            if not csv_file.is_file():
                continue

            split = next(
                (sp for sp, stems in seg_stems.items() if stem in stems), None
            )
            if split is None:
                continue
            frame_file = seg_dirs[split] / f"{stem}.jpg"
            if not frame_file.is_file():
                continue

            records.append({
                "stem": stem,
                "split": split,
                "frame_path": str(frame_file),
                "npy_path": str(npy_file),
                "csv_path": str(csv_file),
            })
        return records

    in_root = Path(in_path.value).expanduser().resolve()
    pair_records = discover(in_root) if in_root.is_dir() else []

    if pair_records:
        pairs_df = pl.DataFrame(pair_records)
        per_split_counts = (
            pairs_df.group_by("split").len(name="n_frames").sort("split")
        )
    else:
        pairs_df = pl.DataFrame(
            schema={
                "stem": pl.String, "split": pl.String,
                "frame_path": pl.String, "npy_path": pl.String,
                "csv_path": pl.String,
            }
        )
        per_split_counts = pl.DataFrame(
            schema={"split": pl.String, "n_frames": pl.UInt32}
        )

    mo.vstack([
        mo.md(
            f"**Input root**: `{in_root}`  \n"
            f"**Frames discovered**: {pairs_df.height}"
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


@app.cell(hide_code=True)
def _(
    CLASS_ID_TO_NAME: dict[int, str],
    OUTPUT_SPLITS,
    consolidate,
    is_empty,
    out_path,
    pairs_df,
    run_button,
    skip_empty_masks,
):
    mo.stop(not run_button.value, mo.md("_Press **Run preprocessing** to execute._"))
    mo.stop(pairs_df.height == 0, mo.md("_No frames discovered — check input path._"))

    def class_ids_from_csv(csv_path: Path) -> list[int]:
        # Each row is a single float; cast to int. Use polars for parsing.
        df = pl.read_csv(csv_path, has_header=False, new_columns=["cls"])
        return [int(round(x)) for x in df["cls"].to_list()]

    def process_frame(
        npy_path: Path, csv_path: Path
    ) -> dict[str, np.ndarray]:
        """Load arrays + class ids, return {class_name: binary uint8 mask}."""
        masks = np.load(npy_path)
        if masks.ndim != 3 or masks.size == 0:
            return {}
        ids = class_ids_from_csv(csv_path)
        n = min(masks.shape[0], len(ids))

        # Group binary masks by class id.
        by_class: dict[int, list[np.ndarray]] = {}
        for i in range(n):
            cls = ids[i]
            if cls not in CLASS_ID_TO_NAME:
                continue
            binary = su.to_uint8_binary(masks[i])
            by_class.setdefault(cls, []).append(binary)

        result: dict[str, np.ndarray] = {}
        for cls, arrs in by_class.items():
            name = CLASS_ID_TO_NAME[cls]
            result[name] = consolidate(arrs) if len(arrs) > 1 else arrs[0]
        return result

    out_root = Path(out_path.value).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    su.reset_split_dirs(out_root, OUTPUT_SPLITS)

    written_records: list[dict] = []
    skipped_frames: list[str] = []
    copied_frames: set[tuple[str, str]] = set()

    rows = list(pairs_df.iter_rows(named=True))
    for row in mo.status.progress_bar(
        rows, title="Writing Frames + Masks", subtitle="frames",
        completion_title="Pipeline complete",
    ):
        stem = row["stem"]
        split = row["split"]
        split_dir = out_root / split
        per_class = process_frame(Path(row["npy_path"]), Path(row["csv_path"]))

        wrote_any = False
        for class_name, arr in per_class.items():
            if skip_empty_masks.value and is_empty(arr):
                continue
            class_dir = split_dir / "Masks" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            with Image.fromarray(arr) as im:
                im.save(class_dir / f"{stem}.png")
            written_records.append({
                "split": split, "class": class_name, "stem": stem,
            })
            wrote_any = True

        if wrote_any and (split, stem) not in copied_frames:
            frames_dir = split_dir / "Frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(row["frame_path"]), frames_dir / f"{stem}.jpg")
            copied_frames.add((split, stem))
        elif not wrote_any:
            skipped_frames.append(stem)

    frame_counts = {
        sp: sum(1 for s, _ in copied_frames if s == sp) for sp in OUTPUT_SPLITS
    }
    mask_counts = {
        sp: sum(1 for r in written_records if r["split"] == sp) for sp in OUTPUT_SPLITS
    }

    mo.md(
        f"**Output root**: `{out_root}`  \n"
        f"**Frames written**: train={frame_counts['train']}, "
        f"val={frame_counts['val']}, test={frame_counts['test']}  \n"
        f"**Class-mask files written**: train={mask_counts['train']}, "
        f"val={mask_counts['val']}, test={mask_counts['test']} "
        f"(total {len(written_records)})  \n"
        f"**Frames skipped** (no class survived filters): {len(skipped_frames)}"
    )
    return out_root, written_records


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation — every mask has a frame and shapes match
    """)
    return


@app.cell(hide_code=True)
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
            f"**Validation passed**: {validation_n} class-masks all paired "
            f"with same-size frames."
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Per-class counts per split
    """)
    return


@app.cell(hide_code=True)
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
    chart = su.class_counts_chart(counts_df, x_title="Class")
    chart
    return


if __name__ == "__main__":
    app.run()
