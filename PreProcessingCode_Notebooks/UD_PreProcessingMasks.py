import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import sys
    import re
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo
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
    # UD (Ureter / Uterine Artery / Nerve) mask preprocessing

    Converts the raw [UD dataset](https://ieee-dataport.org/documents/ud-ureter-uterine-artery-nerve-dataset)
    into the same `Frames/{id}.png` + `Masks/{class}/{id}.png` layout used
    elsewhere in this repo.

    ### Source layout

    ```
    {in}/UD Ureter-Uterine Artery-Nerve Dataset/
        images/  (frames; .png.png for arthery/ureter, .jpg w/ coord suffix for nerve)
        mask/    (per-class binary masks: *_arthery.png, *_ureter.png, *_<x>,<y>_nerve.png)
    ```

    The dataset has asymmetric raw filenames across classes:

    - **arthery + ureter** — frame is `{stem}.png.png`; mask is
      `{stem}.png_arthery.png` or `{stem}.png_ureter.png`.
    - **nerve** — frame is `{stem}.png_<x>,<y>.jpg`; mask is
      `{stem}.png_<x>,<y>_nerve.png`. The `<x>,<y>` is annotation
      metadata (centroid?) that's dropped during the original notebook's
      normalization.

    ### Canonical naming

    The original notebook applied four sequential rename passes (in-place,
    destructive) to collapse all three classes onto a single canonical stem
    `{video}_{sec}_{frac}` derived from `video_<n>.{mov|mp4}_<sec>.<frac>`.
    We replicate that logic as a **pure function**, never mutating source.

    Class typo `arthery` → `artery` is also corrected (the eval CSVs already
    use `artery`).

    ### Splits — driven by existing CSVs

    Splits come from `Datasets_AfterPreprocessing/UD_{train,val,test}.csv`.
    Like Dresden / CholecSeg8k, the split is per-(class, video): the same
    video can land in different splits for different organ classes (e.g.
    `video_20` is in train for artery and test for nerve+ureter). Stems
    not in any CSV are skipped (the original notebook dropped empty masks
    early in its pipeline; the CSVs are the canonical record of what
    survived).

    ### Output layout

    ```
    {out}/{train|val|test}/Frames/{video}_{sec}_{frac}.png
    {out}/{train|val|test}/Masks/{class}/{video}_{sec}_{frac}.png
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


@app.cell
def _():
    # Raw-class suffix → output class name. Fixes the `arthery` typo.
    SRC_CLASS_TO_OUT: dict[str, str] = {
        "arthery": "artery",
        "ureter": "ureter",
        "nerve": "nerve",
    }
    OUTPUT_CLASSES = tuple(sorted(set(SRC_CLASS_TO_OUT.values())))
    OUTPUT_SPLITS = ("train", "val", "test")

    # Extract canonical (video, sec, frac) from raw stems like
    #   "video_14.mov_10.130000"  or  "video_8.mp4_4.555600"
    CANON_RE = re.compile(r"^video_(?P<video>\d+)\.\w+_(?P<sec>\d+)\.(?P<frac>\d+)")

    # Strip a trailing "_<digits>,<digits>" coord suffix.
    COORD_RE = re.compile(r"_\d+,\d+$")
    return CANON_RE, COORD_RE, OUTPUT_SPLITS, SRC_CLASS_TO_OUT


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI controls
    """)
    return


@app.cell
def _():
    in_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_BeforePreprocessing" / "UD Ureter-Uterine Artery-Nerve Dataset"),
        label="Input dataset folder (contains images/ and mask/)",
        full_width=True,
    )
    out_path = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing" / "UD"),
        label="Output dataset folder",
        full_width=True,
    )
    splits_dir = mo.ui.text(
        value=str(PROJECT_ROOT / "Datasets_AfterPreprocessing"),
        label="Folder containing UD_{train,val,test}.csv (canonical splits)",
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
def _(CANON_RE, COORD_RE):
    def canonical_stem_from_raw(body: str) -> str | None:
        """Apply the original notebook's normalization to a raw stem.

        `body` should already have its class suffix and outer extension
        removed. We optionally strip a trailing `_<x>,<y>` coord then
        match the canonical pattern.
        """
        body = COORD_RE.sub("", body)
        m = CANON_RE.match(body)
        if m is None:
            return None
        return f"{m.group('video')}_{m.group('sec')}_{m.group('frac')}"

    def parse_mask_filename(name: str):
        """Return (canonical_stem, src_class) or None.

        Handles both arthery/ureter (no coord) and nerve (coord prefix).
        """
        if not name.endswith(".png"):
            return None
        body = name[:-4]
        for src in ("arthery", "ureter", "nerve"):
            suffix = f"_{src}"
            if body.endswith(suffix):
                inner = body[: -len(suffix)]
                canon = canonical_stem_from_raw(inner)
                return (canon, src) if canon else None
        return None

    def parse_image_filename(name: str):
        """Return canonical_stem or None.

        Handles arthery/ureter (`{stem}.png.png`) and nerve
        (`{stem}.png_<x>,<y>.jpg`) variants.
        """
        if "." not in name:
            return None
        body, _, _ext = name.rpartition(".")
        # Strip an inner trailing ".png" (the .png.png case).
        if body.endswith(".png"):
            body = body[: -len(".png")]
        return canonical_stem_from_raw(body)

    return parse_image_filename, parse_mask_filename


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load split assignments from CSVs

    Each row's `mask` column is `UD/{split}/Masks/{class}/{stem}.png`.
    We extract `(class, video) → split` from those rows.
    """)
    return


@app.cell
def _(splits_dir):
    def load_canonical_splits(splits_root: Path):
        mapping: dict[str, dict[str, str]] = defaultdict(dict)
        per_split_rows: dict[str, int] = {}
        per_stem_split: dict[tuple[str, str], str] = {}
        conflicts: list[tuple[str, str, str, str]] = []
        for split in ("train", "val", "test"):
            csv_path = splits_root / f"UD_{split}.csv"
            if not csv_path.is_file():
                per_split_rows[split] = 0
                continue
            df = pl.read_csv(csv_path)
            per_split_rows[split] = df.height
            mask_col = df["mask"].to_list()
            class_col = df["class_name"].to_list()
            for mask_rel, cls in zip(mask_col, class_col):
                stem = Path(mask_rel).stem
                # stem is "{video}_{sec}_{frac}"
                video = stem.split("_", 1)[0]
                prev = mapping[cls].get(video)
                if prev is not None and prev != split:
                    conflicts.append((cls, video, prev, split))
                mapping[cls][video] = split
                per_stem_split[(stem, cls)] = split
        return dict(mapping), per_split_rows, per_stem_split, conflicts

    splits_root = Path(splits_dir.value).expanduser().resolve()
    canonical_splits, csv_row_counts, stem_class_split, csv_conflicts = (
        load_canonical_splits(splits_root)
        if splits_root.is_dir()
        else ({}, {}, {}, [])
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
        f"**Distinct (class, video) pairs**: {n_class_video}  \n"
        f"**Distinct (stem, class) entries**: {len(stem_class_split)}"
        f"{_conflict_msg}"
    )
    return canonical_splits, stem_class_split


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discover (frame, mask) pairs on disk

    Walks the raw `images/` and `mask/` directories and groups everything
    by canonical stem. Each row is one mask file with its matching frame.
    """)
    return


@app.cell
def _(
    SRC_CLASS_TO_OUT: dict[str, str],
    in_path,
    parse_image_filename,
    parse_mask_filename,
):
    def discover(in_root: Path) -> tuple[list[dict], int, int, int]:
        images_dir = in_root / "images"
        masks_dir = in_root / "mask"
        records: list[dict] = []
        if not images_dir.is_dir() or not masks_dir.is_dir():
            return records, 0, 0, 0

        # Index frames by canonical stem (one stem can have only one image).
        stem_to_image: dict[str, Path] = {}
        skipped_imgs = 0
        for f in sorted(images_dir.iterdir()):
            if not f.is_file() or f.name == "desktop.ini":
                continue
            canon = parse_image_filename(f.name)
            if canon is None:
                skipped_imgs += 1
                continue
            stem_to_image[canon] = f

        skipped_masks = 0
        for m in sorted(masks_dir.iterdir()):
            if not m.is_file() or m.name == "desktop.ini":
                continue
            parsed = parse_mask_filename(m.name)
            if parsed is None:
                skipped_masks += 1
                continue
            canon, src_class = parsed
            class_name = SRC_CLASS_TO_OUT[src_class]
            frame_path = stem_to_image.get(canon)
            if frame_path is None:
                skipped_masks += 1
                continue
            records.append({
                "stem": canon,
                "class_name": class_name,
                "video": canon.split("_", 1)[0],
                "frame_path": str(frame_path),
                "mask_path": str(m),
            })
        return records, len(stem_to_image), skipped_imgs, skipped_masks

    in_root = Path(in_path.value).expanduser().resolve()
    pair_records, n_images, n_skipped_imgs, n_skipped_masks = (
        discover(in_root) if in_root.is_dir() else ([], 0, 0, 0)
    )

    if pair_records:
        pairs_df = pl.DataFrame(pair_records)
        per_class = (
            pairs_df.group_by("class_name").len(name="n_masks").sort("class_name")
        )
        n_stems = pairs_df.select(pl.col("stem").n_unique()).item()
    else:
        pairs_df = pl.DataFrame(
            schema={
                "stem": pl.String, "class_name": pl.String, "video": pl.String,
                "frame_path": pl.String, "mask_path": pl.String,
            }
        )
        per_class = pl.DataFrame(
            schema={"class_name": pl.String, "n_masks": pl.UInt32}
        )
        n_stems = 0

    mo.vstack([
        mo.md(
            f"**Input root**: `{in_root}`  \n"
            f"**Frames discovered**: {n_images} (skipped: {n_skipped_imgs})  \n"
            f"**Mask files matched**: {pairs_df.height} (skipped: {n_skipped_masks})  \n"
            f"**Distinct canonical stems**: {n_stems}"
        ),
        mo.ui.table(per_class, page_size=10),
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
    stem_class_split,
):
    mo.stop(not run_button.value, mo.md("_Press **Run preprocessing** to execute._"))
    mo.stop(pairs_df.height == 0, mo.md("_No frame/mask pairs discovered — check input path._"))
    mo.stop(
        not canonical_splits,
        mo.md("_Canonical splits empty — check splits CSV folder._"),
    )

    out_root = Path(out_path.value).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    su.reset_split_dirs(out_root, OUTPUT_SPLITS)

    written_records: list[dict] = []
    skipped_no_assignment: list[dict] = []
    skipped_empty: list[dict] = []
    copied_frames: dict[tuple[str, str], None] = {}

    cv_to_split = su.invert_class_video_splits(canonical_splits)

    for row in mo.status.progress_bar(
        pairs_df.iter_rows(named=True),
        title="Writing Frames + Masks",
        subtitle="(stem, class) pairs",
        completion_title="Pipeline complete",
        total=pairs_df.height,
    ):
        stem = row["stem"]
        class_name = row["class_name"]
        video = row["video"]
        frame_src = Path(row["frame_path"])
        mask_src = Path(row["mask_path"])

        # Prefer a per-(stem, class) match if the CSV recorded it; otherwise
        # fall back to the per-(class, video) mapping.
        split = stem_class_split.get((stem, class_name)) or cv_to_split.get(
            (class_name, video)
        )
        if split is None:
            skipped_no_assignment.append({
                "stem": stem, "class": class_name, "video": video,
            })
            continue

        binary = su.to_uint8_binary(mask_src)
        if skip_empty_masks.value and not binary.any():
            skipped_empty.append({
                "stem": stem, "class": class_name, "split": split,
            })
            continue

        mask_dir = out_root / split / "Masks" / class_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        with Image.fromarray(binary) as im:
            im.save(mask_dir / f"{stem}.png")

        if (split, stem) not in copied_frames:
            frames_dir = out_root / split / "Frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            # Re-encode as PNG to canonicalize the extension regardless of
            # whether the source was .png.png or .jpg.
            with Image.open(frame_src) as fim:
                fim.convert("RGB").save(frames_dir / f"{stem}.png")
            copied_frames[(split, stem)] = None

        written_records.append({
            "split": split, "class": class_name, "stem": stem, "video": video,
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
        f"**Empty masks skipped**: {len(skipped_empty)}"
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

    validation_issues, validation_n = su.validate_frames_masks_paired(
        out_root, OUTPUT_SPLITS, frame_ext="png",
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
    chart = su.class_counts_chart(counts_df, width=520)
    chart
    return


if __name__ == "__main__":
    app.run()
