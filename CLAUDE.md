# photo-organizer

Single-file CLI tool (`photo_organizer.py`) that scans a folder of photos, extracts
the most reliable date for each (EXIF → filename → filesystem, in that priority),
finds near-duplicates via perceptual hashing, and copies (never moves/deletes)
everything into `output/organized/YYYY/MM/` plus `output/duplicates/`, with an
HTML report at the end.

## Layout

- `photo_organizer.py` — the tool (~700 lines). Photos are passed around as a
  `PhotoRecord` dataclass (`path`, `size`, `date`, `date_source`, `file_hash`,
  `phash`, `error`); pipeline results use an `OrganizeResults` dataclass.
- `tests/test_photo_organizer.py` — pytest suite covering date parsing,
  `find_duplicates` grouping, `safe_copy` collision/overwrite handling, symlink
  skipping in `scan_photos`, and `validate_paths`.
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
- `find_duplicates()` — exact match via MD5, then O(n²) perceptual-hash clustering
  (`--hash-threshold`, default 4) for photos not already exact-matched. Prints a
  warning above `VISUAL_DEDUP_WARN_THRESHOLD` (50k) photos but still proceeds.
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

## Remaining known tradeoffs (not bugs)

- Broad `except Exception: pass` blocks throughout intentionally keep a single
  bad file from crashing the whole run — expected tradeoff for this tool, not
  something to "fix" reflexively.
- `find_duplicates`'s visual clustering is still O(n²); the warning threshold is
  a stopgap, not an algorithmic fix — revisit if runs on >50k-photo libraries
  become common. Measured (2026-07-03, isolated benchmark of the comparison
  loop, not full pipeline): 500 photos → 1.6s, 2,000 → 27.9s, 10,000 → 633.6s
  (~10.6 min) — a near-exact n² fit. Extrapolates to ~43 min at 20k and several
  hours at 50k. This step doesn't parallelize with `--workers`.

## Conventions to preserve

- Never move or delete source files — everything is copy-only, matching the
  README's explicit guarantee.
- `--dry-run` must remain a true no-op on the filesystem (only report generation
  writes anything).
- Keep `safe_copy`'s `O_EXCL` atomic-create pattern when touching copy logic —
  it's the one place a race condition would cause silent data loss.
