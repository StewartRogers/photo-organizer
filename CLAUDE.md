# photo-organizer

Single-file CLI tool (`photo_organizer.py`) that scans a folder of photos, extracts
the most reliable date for each (EXIF → filename → filesystem, in that priority),
finds near-duplicates via perceptual hashing, and copies (never moves/deletes)
everything into `output/organized/YYYY/MM/` plus `output/duplicates/`, with an
HTML report at the end.

## Layout

- `photo_organizer.py` — the tool (~800 lines). Photos are passed around as a
  `PhotoRecord` dataclass (`path`, `size`, `date`, `date_source`, `file_hash`,
  `phash`, `error`); pipeline results use an `OrganizeResults` dataclass.
- `tests/test_photo_organizer.py` — pytest suite covering date parsing,
  `find_duplicates` grouping (including the BK-tree-backed visual clustering),
  `safe_copy` collision/overwrite handling, symlink skipping in `scan_photos`,
  `validate_paths`, `resolve_log_path`/`setup_logging`, and the `errors/`-folder
  copy path in `organize_photos`.
- `requirements.txt` / `requirements-dev.txt` — runtime deps (Pillow, piexif,
  imagehash, tqdm, colorama, pillow-heif) and dev deps (adds pytest),
  respectively. No other build/lint config — plain `python3 photo_organizer.py
  ...`; tests via `pytest tests/` (`pip install -r requirements-dev.txt`
  first).

## Key functions

- `get_photo_date()` — date-priority logic (EXIF DateTimeOriginal → Digitized →
  DateTime → filename pattern → filesystem mtime/ctime).
- `scan_photos()` — `os.walk(..., followlinks=False)`-based recursive scan that
  explicitly skips symlinked files and directories (security fix, see below).
- `find_duplicates()` — exact match via MD5, then perceptual-hash clustering
  (`--hash-threshold`, default 4) for photos not already exact-matched, using a
  `BKTree` (see below) instead of all-pairs comparison. Prints a warning above
  `VISUAL_DEDUP_WARN_THRESHOLD` (50k) photos but still proceeds.
- `BKTree`/`_BKNode` — a BK-tree (metric tree keyed on Hamming distance) that
  answers "all photos within `hash_threshold` of this hash" without comparing
  against every other photo. `find_duplicates()`'s greedy leader/claim loop is
  unchanged — the tree only changes how fast each lookup is, not which matches
  it finds (relies on the triangle inequality holding for Hamming distance).
- `safe_copy()` — copies using `os.open(..., O_CREAT | O_EXCL)` to atomically avoid
  overwriting an existing destination file; appends `_1`, `_2`, ... on collision.
- `process_all_photos()` / `organize_photos()` — the parallel-processing and
  copy-loop steps extracted out of `main()`.
- `build_report_html()` / `generate_report()` — HTML templating is separated from
  the atomic-write-to-disk step.
- `validate_paths()` — resolves `--source`/`--output` to absolute paths and
  rejects configs where one is nested inside the other.
- `main()` — now a thin orchestrator: parse args → validate paths → scan →
  process → dedup → organize → report → print summary.

## Fixed in the 2026-07-03 review pass

- **Symlink following (medium security):** `scan_photos()` now skips symlinked
  files/dirs so a symlink inside `--source` can't cause files outside the
  intended tree to be read and copied into `--output`.
- **Unvalidated `--output` (low security):** `validate_paths()` now resolves
  both paths and exits with an error if `--output` is nested inside `--source`
  or vice versa.
- Removed dead `errno`/`tempfile` imports; replaced magic numbers (`65536`
  chunk size, `500`/`200`/`100` report row caps) with named constants.
- Replaced the dict-as-record pattern with `PhotoRecord`/`OrganizeResults`
  dataclasses.
- Split `main()` and `generate_report()` into single-purpose functions
  (`scan_photos`, `process_all_photos`, `organize_photos`, `build_report_html`,
  `generate_report`, `print_summary`).
- Added the `tests/` suite (previously zero coverage).
- Added file-based logging (`setup_logging()`, `--log-file`, default
  `<output>/photo_organizer.log`): per-photo processing errors and copy
  failures are logged, and `logging.captureWarnings(True)` routes Python
  warnings (e.g. Pillow's palette/transparency `UserWarning`) into the log
  file instead of stderr, so they don't interleave with/garble the `tqdm`
  progress bars.
- Root-caused the Pillow "Palette images with Transparency..." `UserWarning`:
  `compute_perceptual_hash()` now converts `P`-mode-with-transparency images
  to `RGBA` before `RGB` (Pillow's own recommended path), instead of relying
  solely on the log-file safety net above.
- The HTML report's Errors table was silently truncating past
  `REPORT_MAX_ERROR_ROWS` (200) with no indication rows were cut, unlike the
  Suspicious Dates table. Fixed to show the same `"...and N more"` overflow
  note.
- Photos that fail `process_photo()` (corrupt file, unreadable EXIF, hashing
  failure, etc.) used to be dropped from `organize_photos()` entirely — never
  copied anywhere, only mentioned in the report. They're now copied to a new
  `output/errors/` folder (via the existing `safe_copy()`, which doesn't touch
  PIL/piexif/imagehash and so usually still works) so there's always a
  physical copy to inspect. `OrganizeResults.errors_copied` tracks the count.
- `find_duplicates`'s visual clustering was O(n²) all-pairs comparison;
  replaced with a `BKTree`-based lookup (see Key Functions) that finds the
  same matches without comparing every photo against every other photo.

## Remaining known tradeoffs (not bugs)

- Broad `except Exception: pass` blocks throughout intentionally keep a single
  bad file from crashing the whole run — expected tradeoff for this tool, not
  something to "fix" reflexively.
- `find_duplicates`'s visual clustering now uses a `BKTree` instead of all-pairs
  comparison, which is a large average-case improvement, but a BK-tree's
  worst case can still degrade toward O(n²) for pathological hash
  distributions or a very large `--hash-threshold` (the tree gets little
  benefit from the triangle-inequality pruning when the radius is close to
  the hash's bit length). `VISUAL_DEDUP_WARN_THRESHOLD` (50k) still prints an
  advisory rather than gating anything. This step doesn't parallelize with
  `--workers`. Re-benchmark before assuming a specific runtime at very large
  scale or unusually high thresholds.

## Conventions to preserve

- Never move or delete source files — everything is copy-only, matching the
  README's explicit guarantee.
- `--dry-run` must remain a true no-op on the filesystem (only report generation
  writes anything).
- Keep `safe_copy`'s `O_EXCL` atomic-create pattern when touching copy logic —
  it's the one place a race condition would cause silent data loss.
