# photo-organizer

Single-file CLI tool (`photo_organizer.py`) that scans a folder of photos and
videos, extracts the most reliable date for each (EXIF → filename → filesystem,
in that priority; videos skip straight to filename → filesystem since Pillow
can't read video containers), finds near-duplicates via perceptual hashing
(photos only — videos are never deduplicated), and copies (never moves/deletes)
everything into `output/organized/YYYY/MM/` plus `output/duplicates/`, with an
HTML report at the end.

## Layout

- `photo_organizer.py` — the tool (~950 lines). Photos/videos are passed around
  as a `PhotoRecord` dataclass (`path`, `size`, `date`, `date_source`,
  `file_hash`, `phash`, `error`, `cloud_only`); pipeline results use an
  `OrganizeResults` dataclass.
- `tests/test_photo_organizer.py` — pytest suite covering date parsing,
  `find_duplicates` grouping (including the BK-tree-backed visual clustering),
  `safe_copy` collision/overwrite handling, symlink skipping in `scan_photos`,
  `validate_paths`, `resolve_log_path`/`setup_logging`, the `errors/`-folder
  copy path in `organize_photos`, cloud-only-placeholder detection, the
  `retry_photos_<run_id>.txt` round-trip, and video scanning/dating/no-dedup
  behavior.
- `requirements.txt` / `requirements-dev.txt` — runtime deps (Pillow, piexif,
  imagehash, tqdm, colorama, pillow-heif) and dev deps (adds pytest),
  respectively. No other build/lint config — plain `python3 photo_organizer.py
  ...`; tests via `pytest tests/` (`pip install -r requirements-dev.txt`
  first).

## Key functions

- `get_photo_date()` — date-priority logic (EXIF DateTimeOriginal → Digitized →
  DateTime → filename pattern → filesystem mtime/ctime).
- `get_video_date()` — the video equivalent: container `creation_date` metadata
  (via `hachoir`, if `VIDEO_METADATA_SUPPORTED`) → filename pattern →
  filesystem mtime/ctime. Mirrors `get_photo_date()`'s tier structure
  (including preferring filesystem over a *suspicious* primary-tier date, but
  falling back to that suspicious date labeled as such if nothing else is
  available) but is a separate function rather than a shared refactor —
  deliberate, to avoid touching `get_photo_date()`'s already-tested behavior.
  `process_photo()` calls this instead of `get_photo_date()` when `is_video()`.
- `is_video()` — extension check against `SUPPORTED_VIDEO_EXTENSIONS`. Used by
  `process_photo()` both to route to `get_video_date()` and to skip
  `compute_file_hash()`/`compute_perceptual_hash()` for videos entirely
  (rather than compute-then-ignore) — leaving `file_hash`/`phash` unset
  (`None`) is what makes `find_duplicates()` (which only groups records with a
  truthy hash) naturally never consider videos, with no special-casing needed
  in `find_duplicates()` or `organize_photos()`.
- `scan_photos()` — `os.walk(..., followlinks=False)`-based recursive scan that
  explicitly skips symlinked files and directories (security fix, see below).
  Matches both `SUPPORTED_PHOTO_EXTENSIONS` and `SUPPORTED_VIDEO_EXTENSIONS`.
- `find_duplicates()` — exact match via MD5, then perceptual-hash clustering
  (`--hash-threshold`, default 4) for photos not already exact-matched, using a
  `BKTree` (see below) instead of all-pairs comparison. Prints a warning above
  `VISUAL_DEDUP_WARN_THRESHOLD` (50k) photos but still proceeds. Videos never
  appear here (see `is_video()` above).
- `BKTree`/`_BKNode` — a BK-tree (metric tree keyed on Hamming distance) that
  answers "all photos within `hash_threshold` of this hash" without comparing
  against every other photo. `find_duplicates()`'s greedy leader/claim loop is
  unchanged — the tree only changes how fast each lookup is, not which matches
  it finds (relies on the triangle inequality holding for Hamming distance).
- `safe_copy()` — copies using `os.open(..., O_CREAT | O_EXCL)` to atomically avoid
  overwriting an existing destination file; appends `_1`, `_2`, ... on collision.
  Returns `(path, None)` on success or `(None, error_message)` on failure — the
  caller always has the real reason, never just a bare `None`.
- `is_cloud_only_placeholder()` — Windows-only check (`st_file_attributes &
  FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`) for cloud-sync placeholder files (e.g.
  OneDrive Files On-Demand) that haven't been downloaded locally yet. `stat()`
  succeeds on these but reading content fails, so `process_photo()` checks this
  first and short-circuits with `PhotoRecord.cloud_only = True` instead of
  hitting a confusing `[Errno 22] Invalid argument` from the OS.
- `process_all_photos()` / `organize_photos()` — the parallel-processing and
  copy-loop steps extracted out of `main()`. `organize_photos()` treats
  cloud-only photos as a third bucket alongside organized/duplicate/error:
  left untouched in `--source` (nothing to copy), reported separately, and
  excluded from both the `errors` list and the `suspicious_dates` list (they
  never reach `get_photo_date()`, so `date is None` would otherwise
  incorrectly flag every one of them as suspicious too).
- `write_retry_file()` / `load_retry_paths()` — write/read
  `<output>/retry_photos_<run_id>.txt` (errors + cloud-only paths, one per
  line). Each run gets its own timestamped file rather than overwriting a
  shared name, so a history builds up and you choose which run's list to feed
  back via `--retry-file`.
- `estimate_required_bytes()` / `check_disk_space()` — pre-flight worst-case
  destination-space check, run right after scanning (before the expensive
  process/hash pass). Sums `stat().st_size` over every scanned path — this
  works on cloud-only placeholders without downloading them, so their bytes
  are reported separately (not counted, since they're skipped this run). The
  "worst case" is close to exact: `organize_photos()` copies duplicates too
  (to `duplicates/`), so total destination usage barely depends on the
  duplicate ratio. Exits before processing starts if space is short, unless
  `--dry-run` (warns only, since dry runs never copy anything).
- `build_report_html()` / `generate_report()` — HTML templating is separated from
  the atomic-write-to-disk step.
- `validate_paths()` — resolves `--source`/`--output` to absolute paths and
  rejects configs where one is nested inside the other.
- `main()` — now a thin orchestrator: parse args → validate paths → scan (or
  load `--retry-file`) → process → dedup → organize → write retry file →
  report → print summary.

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

## Fixed in the 2026-07-04 follow-up (OneDrive incident)

A 52,687-photo overnight run against a OneDrive-backed `--source` logged 14,972
identical `[Errno 22] Invalid argument` errors (both on initial processing and
on the `errors/`-folder safety-net copy). Root cause: those files were OneDrive
Files On-Demand cloud-only placeholders never downloaded to the machine —
confirmed by checking `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS` on the actual
failing paths, and reproduced with a plain single-threaded `open()` outside the
tool entirely (ruling out `--workers` concurrency as the cause).

- Added `is_cloud_only_placeholder()` and `PhotoRecord.cloud_only` (see Key
  Functions) so these are detected up front and reported as their own category
  instead of a generic, misleading error.
- `safe_copy()` changed from returning bare `Optional[Path]` to `(path, error)`
  — previously, a failed `errors/`-folder copy logged the *original* processing
  error again instead of the real reason the second copy attempt failed.
- Added `write_retry_file()`/`load_retry_paths()` and `--retry-file` so a
  follow-up run (e.g. after OneDrive finishes syncing) can reprocess just the
  photos that didn't make it in, instead of a full re-scan.
- `organize_photos()`'s `suspicious_dates` list now excludes `cloud_only`
  photos — they never reach `get_photo_date()` so `date is None`, which was
  otherwise flagging all 14,972 of them as "suspicious" too, on top of already
  being reported as cloud-only.
- A later, separate run against the same library hit `[Errno 28] No space
  left on device` partway through copying (6 files, ~2 hours into the run).
  Added `estimate_required_bytes()`/`check_disk_space()` (see Key Functions)
  as a pre-flight gate so this is caught in seconds, before the expensive
  process/hash pass, instead of hours in.

## Added in the 2026-07-04 video-support pass

Videos are now scanned and organized alongside photos, by explicit request —
copy-only (matching photos), and **intentionally never deduplicated** (user's
call: "no need to do any duplicate checks" for videos).

- Added `SUPPORTED_VIDEO_EXTENSIONS` and `is_video()`; `scan_photos()` now
  matches photo or video extensions. `SUPPORTED_EXTENSIONS` was renamed to
  `SUPPORTED_PHOTO_EXTENSIONS` for clarity (no behavior change for photos).
- `process_photo()` skips `compute_file_hash()`/`compute_perceptual_hash()`
  for videos — see `is_video()` in Key Functions for why that alone is
  sufficient to keep them out of `find_duplicates()` with no other changes.
- First cut used filename/filesystem dating only for videos (deferred reading
  container metadata to avoid a new dependency). Follow-up (same day, user
  asked explicitly): added `get_video_date()` + the `hachoir` dependency to
  read a video container's own `creation_date` metadata (MP4/MOV/AVI/MKV/...)
  as the first-tier source — the video equivalent of EXIF `DateTimeOriginal`
  — before falling back to filename → filesystem, mirroring
  `get_photo_date()`'s tier structure exactly (including "remember a
  suspicious primary-tier date, prefer filesystem, but return it labeled
  SUSPICIOUS as a last resort" behavior). `hachoir` is a soft dependency like
  `pillow_heif`/HEIC: `VIDEO_METADATA_SUPPORTED` flag, warns once at startup
  if missing, falls back to filename/filesystem gracefully rather than
  crashing. Verified against a real MP4 (`hachoir.core.config.quiet = True`
  is needed to silence its own console warning spam on unparseable files;
  `parser.stream.close()` after `extractMetadata()` to avoid leaking file
  handles across thousands of videos in the thread pool).

## Remaining known tradeoffs (not bugs)

- Videos are never deduplicated by design (see above) — genuinely duplicate
  video files (e.g. the same clip backed up twice) will both land in
  `organized/`, unlike photos. Not a bug; revisit only if asked.
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
