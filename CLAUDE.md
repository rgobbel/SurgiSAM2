# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goals

### Overall goal
- Assess the practicality of running segmentation locally rather than in the cloud.

### Immediate goals
1. Convert messy Jupyter notebooks in `/home/gobbel/src/SurgiSAM2/` and its subdirectories into clean, idempotent Marimo notebooks.
2. Run SAM2 segmentation

## First notebook to tackle
`PreProcessingCode_Notebooks/m2caiSegDataset_PreProcessingMasks.ipynb`

## Design decisions already made

- Split by image, not by mask, to prevent train/test leakage (same image's masks for different organ classes were landing in different splits in the original).
- Keep per-color binary mask extraction for now (one binary PNG per organ class per image).
- Make the notebook idempotent — re-running on already-processed data is safe.
- Class colors → snake_case names, no spaces or punctuation in file/folder names. Use lowercase hex (e.g. 55aaff) only if a color literal must appear in a path. The class color → name map is in the original notebook (cell defining color_to_class_name), with these corrections:
  - `gall-bladder` → `gall_bladder` `(color (85, 170, 255))` — was incorrectly dropped from `allowed_colors` in the original; must be preserved.
  - `specimen-bag` → `specimen_bag`
- UI inputs for `Datasets_BeforePreprocessing/m2caiSeg` and `Datasets_AfterPreprocessing/m2caiSeg` paths (defaulted to current values).
- Use polars, not pandas, for any tabular work.
- Output layout per split: `{out}/{train|val|test}/Frames/{image_id}.jpg` and `{out}/{train|val|test}/Masks/{class_name}/{image_id}.png`

## Known bugs in the original to fix

- Step 12 (frame redistribution after reshuffle) references a TempFrames directory that's never created — silently fails, leaves splits empty.
- Step 11 (mask reshuffle) splits per-organ-class independently → same image lands in multiple splits.
- Color-folder generation in step 6 writes per-color subdirs inside the masks dir alongside the original masks.
- Step 15 hardcodes mismatch records pasted from a previous run's output — not reproducible.

## Other goals

- Replace the 6+ duplicated for split, folder_path in ... loops with one helper.
- Use pathlib.Path throughout.
- Use context managers everywhere (the original was inconsistent — sometimes used with Image.open(...), sometimes not).
- Add a validation cell that confirms every mask has a corresponding frame and shapes match.
- Add an Altair chart of per-class counts per split.

## Suggested cell structure (~10 cells)

- Imports
- Constants (CLASS_COLORS dict, helper rgb_to_hex)
- UI controls (paths, thresholds, seed, train/val fractions, run button)
- Pure functions (get_color_counts, filter_small_colors, extract_color_mask, largest_component_area)
- Discover input frames + masks
- Run-button-gated pipeline: filter → extract → connected-component filter → split by image → write outputs
- Validation: shape parity, frame/mask correspondence
- Per-class count table (mo.ui.table of polars DataFrame)
- Altair bar chart of class counts per split

1After this notebook is done, repeat the pattern for the other notebooks in `PreProcessingCode_Notebooks/`. Look for shared logic to extract into a `surgisam_utils.py` module rather than duplicating across notebooks.

