# photo-organizer

# 📷 Photo Organizer & Deduplicator

Scans tens of thousands of photos (and videos), removes near-duplicates, and organizes everything into clean `YYYY/MM/` folders — all without touching your originals.

---

## What It Does

1. **Scans** your source folder recursively for all image *and video* files
2. **Checks destination free space** before doing any real work — see below
3. **Extracts the most reliable date** from each photo using this priority:
   - EXIF `DateTimeOriginal` ← most trustworthy (set by camera at capture)
   - EXIF `DateTimeDigitized`
   - EXIF `DateTime` (can be modified by editors — used as fallback)
   - Filename pattern (e.g. `IMG_20181215_134500.jpg`)
   - File system date ← least trustworthy
   - Videos use the same filename/file-system tiers (no EXIF step — see below)
4. **Flags suspicious dates** — years outside 1990–present are noted in the report
5. **Detects duplicates** using perceptual hashing (very strict: catches re-saves, crops, minor edits) — **photos only**, see below
6. **Keeps the best-quality copy** when duplicates are found — highest resolution, then one with a real EXIF date, then largest file, then oldest date
7. **Copies** organized photos and videos to `output/organized/YYYY/MM/`, together by date — camera RAW files (NEF, CR2, DNG, ...) and TIFFs go into an `originals/` subfolder there and are never treated as duplicates
8. **Copies duplicates** to `output/duplicates/` for your review
9. **Generates an HTML report** with all actions, suspicious dates, and duplicate groups
10. **Optionally archives `--source`** into chunked, verified zip files after a
    fully successful run (`--archive-source`) — see below

> ✅ Your originals are **never moved or deleted**. Everything is copied.

---

## Requirements

### Python
Python **3.12, 3.13, 3.14 or 3.15**. Any of these works, and the tool is
tested against all four on every push (see `.github/workflows/tests.yml`).
3.11 and older are no longer supported. (3.9 and 3.10 were dropped because
the Pillow release that fixes the October 2026 image-decoder security
advisories, 13.0.0, needs Python 3.11 or newer; 3.11 was dropped later when
the floor moved to 3.12.)
Download from: https://www.python.org/downloads/

Running on an older interpreter exits immediately with a clear message rather
than failing partway through a run.

### Install Dependencies (in a virtual environment)

Install into a project-local virtual environment (`.venv\`, already in
`.gitignore`) so these packages don't mix with other Python projects. On
Windows, from the project folder in PowerShell:

```
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(Any supported version works in place of `-3.15`; `py -0` lists the ones
installed. In Command Prompt, activate with `.venv\Scripts\activate.bat`
instead. If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

Activate it again in each new terminal before running the tool. Once it's
active, `python` means the venv's Python, so the commands below work as
written. On macOS/Linux, create it with `python3 -m venv .venv` and activate
it with `source .venv/bin/activate`.

This installs Pillow, piexif, imagehash, tqdm, colorama, pillow-heif (needed
for iPhone HEIC photos — without it, HEIC files are scanned but skipped), and
hachoir (needed to read video creation dates — without it, videos fall back
to filename/filesystem dating).

---

## Usage

### Basic (recommended first run — dry run preview)

```
python photo_organizer.py --source "C:\Photos\My Pictures" --output "C:\Photos\Organized" --dry-run
```

This scans everything and generates a report **without copying any files**.  
Open `C:\Photos\Organized\photo_organizer_report.html` to review what will happen.

---

### Apply for real

```
python photo_organizer.py --source "C:\Photos\My Pictures" --output "C:\Photos\Organized"
```

---

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | required | Folder containing your photos (searched recursively) |
| `--output` | required | Where to put organized photos + report |
| `--dry-run` | off | Preview only — no files are copied |
| `--hash-threshold` | 4 | Visual similarity strictness (0 = identical only, 4 = very strict, 10 = moderate) |
| `--workers` | 4 | Parallel threads for hashing (increase to 8 on fast machines) |
| `--log-file` | `<output>/photo_organizer.log` | Path to the log file (errors and library warnings) |
| `--retry-file` | none | Reprocess only the photos listed in this file (one path per line) instead of rescanning `--source` — see `<output>/retry_photos.txt` from a previous run |
| `--max-cloud-only-gb` | 1.0 | Abort before processing if more than this many GB of source files are still cloud-only / not downloaded yet |
| `--archive-source` | off | After a fully successful run, zip `--source` into chunks (written to `--source`'s own root) |
| `--archive-chunk-gb` | 1.0 | Target size per zip chunk when using `--archive-source` or `--archive-only` |
| `--archive-only` | off | Skip scanning/organizing entirely and just archive `--source` now (e.g. forgot `--archive-source` earlier) |

---

## Output Structure

```
C:\Photos\Organized\
├── organized\
│   ├── 2018\
│   │   ├── 06\
│   │   │   ├── IMG_1234.jpg
│   │   │   ├── DSC_0042.jpg
│   │   │   └── originals\  ← camera RAW + TIFF files (never deduplicated)
│   │   │       ├── DSC_0042.NEF
│   │   │       └── DSC_0042.tif
│   │   └── 12\
│   ├── 2019\
│   └── unknown_date\       ← photos with no usable date
├── duplicates\              ← review these before deleting
│   ├── IMG_1234_copy.jpg
│   └── ...
├── errors\                  ← photos that failed date/hash processing, copied here for inspection
│   └── ...
├── photo_organizer_report.html
├── photo_organizer.log                    ← errors and library warnings, appended across runs
└── retry_photos_20260704_101423.txt       ← errors + cloud-only files from *this* run only, for --retry-file
```

Photos that are cloud-only placeholders (e.g. OneDrive Files On-Demand files not yet
downloaded to this machine) are handled one of two ways — see "Pre-Flight Disk
Space Check" above. If the backlog is small, they're downloaded automatically
and organized normally. Otherwise (or under `--dry-run`) they're **left
untouched in `--source`** — not copied to `errors/`, since there's nothing to
read yet — and listed separately in the report and in that run's
`retry_photos_<timestamp>.txt`.

Each run writes its own `retry_photos_<timestamp>.txt` (named after when that run
started) rather than overwriting a single shared file — so a history of retry
lists builds up across runs and you can pick which one to hand to `--retry-file`
(e.g. an older run's list if a later run's retry attempt itself hit new problems).

---

## Pre-Flight Disk Space Check

Before extracting dates or computing hashes (the slow, potentially hours-long
step), the tool sizes every scanned file with a plain `stat()` call — this
works even on cloud-only placeholders (no download triggered) — and compares
the total against free space on the `--output` drive. This estimate is a
worst case, but a fairly exact one: duplicates are still copied (to
`duplicates/` instead of `organized/`), so total destination usage is
essentially the full size of everything readable, regardless of how many
turn out to be duplicates. Cloud-only files aren't copied this run, so their
size is reported separately and not counted against the requirement.

If there isn't enough free space, the tool exits immediately with the
shortfall in GB rather than discovering it hours into copying (this is
exactly what happened before this check existed — see `CLAUDE.md`). Under
`--dry-run` this only warns, since a dry run never copies anything anyway.

The same pre-flight pass also checks how much of the source is still
cloud-only:

- **Over `--max-cloud-only-gb`** (default 1.0 GB): the tool exits and asks
  you to let the sync catch up before trying again — running against a
  library that's still mostly cloud-only would barely organize anything, and
  only widens the window where duplicate detection can't see across separate
  runs (see "Reprocessing" below). Raise the threshold if running against a
  partially-synced library is expected. Under `--dry-run` this only warns.
- **Under the threshold**: the backlog is small enough to just deal with
  now — any cloud-only file encountered during processing is actively
  downloaded (the same "read triggers hydration" mechanism OneDrive itself
  uses) and folded into this run instead of being skipped for a future
  `--retry-file` pass. This never happens under `--dry-run` (a dry run must
  never touch the filesystem, and triggering a download counts).

---

## Archiving the Source (`--archive-source`)

After a run with **zero errors and zero cloud-only leftovers**, `--archive-source`
packs everything under `--source` into a sequence of independent zip files
(~`--archive-chunk-gb` each, default 1.0 GB), written directly into
**`--source`'s own root** — e.g. a 38 GB library becomes roughly 38
`archive_<run_id>_NNNN.zip` files sitting alongside your original photos.

- **Your originals are untouched** — archiving only adds new `.zip` files; it
  never deletes or modifies anything already in `--source`, consistent with
  this tool's copy-only guarantee everywhere else.
- **No compression** (`ZIP_STORED`) — photos and videos are already
  compressed formats, so deflate would mostly just spend CPU without
  shrinking the output. This is chunking for easier transfer/upload, not
  space-saving.
- **A single file larger than the chunk size still gets its own (larger)
  chunk** — individual files are never split across archives.
- **Every chunk is verified immediately after being written**, using
  `zipfile`'s own CRC check (`testzip()`) — not just "did it open," but "does
  every entry's content actually match what was written." Any corrupt or
  unreadable chunk is reported, not silently trusted.
- Re-running `--archive-source` later won't re-zip its own previous output —
  existing `archive_*.zip` files are skipped when scanning `--source`.
- Skipped (with a message, not silently) under `--dry-run`, or if the run had
  any errors or cloud-only files left over — resolve those and re-run first.

**Forgot to pass `--archive-source` on a run that already finished successfully?**
Use `--archive-only` instead of re-running the whole scan/organize pipeline —
it skips straight to archiving `--source` with the same chunking/verification
behavior described above. `--output` is still required (it's used for the log
file) but nothing is written there. Also skipped under `--dry-run`.

```
python photo_organizer.py --source "C:\Photos\My Pictures" --output "C:\Photos\Organized" --archive-only
```

---

## Video Handling

Video files (`.mp4`, `.mov`, `.avi`, `.mpg`, `.mpeg`, `.m4v`, `.wmv`, `.3gp`,
`.mkv`, `.flv`, `.webm`) are scanned and organized right alongside photos, so a
trip's photos and video clips end up together in the same `output/organized/YYYY/MM/`
folder. Two differences from photo handling:

- **Dating**: videos have no EXIF, but MP4/MOV/AVI/MKV/... containers carry
  their own `creation_time` metadata — the video equivalent of EXIF
  `DateTimeOriginal` — read via [hachoir](https://pypi.org/project/hachoir/).
  Priority is: container metadata → filename pattern (e.g.
  `VID_20190704_120000.mp4`) → file system date. If `hachoir` isn't installed,
  the tool warns once at startup and falls back to filename/filesystem only.
  MP4/MOV/M4V/3GP, MKV/WebM and WMV store that time in **UTC**, so it's
  converted to this computer's local time zone — otherwise an evening clip
  near the end of a month could land in the next month's folder, away from
  the photos taken alongside it. AVI/FLV/MPEG dates are used as-is.
- **No deduplication**: videos are never hashed and never compared for
  duplicates — they always land in `organized/`, never `duplicates/`. If you
  have genuinely duplicate video files, this tool won't catch them.

---

## Understanding the Report

The HTML report (open in any browser) contains:

- **Summary stats** — totals, time taken
- **Suspicious dates** — photos where the date might be wrong (out-of-range years, no EXIF data)
- **Duplicate groups** — which file was kept and which were moved to `/duplicates`
- **Errors** — any files that couldn't be fully processed (also copied to `/errors` for inspection)
- **Cloud-only (not downloaded)** — files whose content isn't available locally yet (e.g. OneDrive Files On-Demand placeholders), so they couldn't be read; nothing was copied for these

---

## Reprocessing Failed / Not-Yet-Synced Photos

If a run has errors or cloud-only files, an `<output>/retry_photos_<timestamp>.txt`
file is written listing just those paths (one file per run — see Output Structure
above). Once you've fixed the underlying issue (e.g. let OneDrive finish syncing —
see below), rerun with:

```
python photo_organizer.py --source "C:\Photos\My Pictures" --output "C:\Photos\Organized" --retry-file "C:\Photos\Organized\retry_photos_20260704_101423.txt"
```

This reprocesses only those photos instead of rescanning your whole library.

### A note on OneDrive / cloud-sync folders

If `--source` lives under OneDrive (or Dropbox/Google Drive) with "Files On-Demand"
enabled, some files may be cloud-only placeholders that haven't been downloaded to
this machine — reading them fails until they're synced, even though they show up
in a normal folder listing. This tool detects and reports these separately rather
than treating them as errors. To fix: make sure OneDrive is running and signed in,
and either wait for background sync to finish or right-click the folder →
"Always keep on this device" to force download, then re-run (ideally with
`--retry-file`).

---

## Notes on Safety

- `--output` must not be nested inside `--source` (or vice versa) — the tool checks this on startup and exits with an error rather than risk scanning its own output or writing over your source photos.
- Symlinked files and folders inside `--source` are skipped, so a stray symlink pointing outside your photo library can't cause unrelated files to be read or copied.

## Running Tests

With the virtual environment active (see Install Dependencies):

```
pip install -r requirements-dev.txt
pytest tests/
```

To check the code still runs on the oldest supported Python (3.12) without
having 3.12 installed:

```
vermin --target=3.12- --no-tips photo_organizer.py tests/
```

This is a static check — it catches syntax that is only legal on a newer
interpreter (the most likely way support silently breaks when the project is
edited from machines running different Python versions).

To check the dependencies for known security advisories (CI runs this too):

```
pip-audit -r requirements.txt
```

---

## Supported Formats

**Photos** (dated, deduplicated): JPG, JPEG, PNG, GIF, BMP, WebP, HEIC/HEIF (with pillow-heif)

**Originals: camera RAW and TIFF** (dated, *not* deduplicated, copied to `YYYY/MM/originals/`): CR2, CR3, NEF, ARW, ORF, RW2, DNG, RAF, PEF, SRW, RAW, TIFF/TIF. EXIF dates are read from TIFF files and the TIFF-based RAW formats (NEF, CR2, DNG, ARW, PEF, SRW); CR3, RAF, ORF and RW2 aren't TIFF files, so they're dated from filename → file system date instead. Because TIFFs are never compared, duplicate TIFFs (e.g. the same scan saved twice) are both kept

**Videos** (dated, *not* deduplicated — see [Video Handling](#video-handling)): MP4, MOV, AVI, MPG, MPEG, M4V, WMV, 3GP, MKV, FLV, WebM

---

## Tips for 10,000+ Photos

- **Run `--dry-run` first** — always. The report will show you suspicious dates before anything moves.
- **Use `--workers 8`** on a modern machine to speed up date/hash extraction — note
  this does **not** speed up the visual-duplicate comparison step below, which is
  single-threaded.
- The visual-duplicate comparison step uses a BK-tree index (instead of comparing
  every photo against every other photo), so it scales far better than a naive
  approach — tens of thousands of photos should take minutes, not hours. Very
  large libraries or an unusually high `--hash-threshold` can still be slower in
  the worst case, so budget extra time and prefer to run unattended for the
  first pass on a new library.
- After running, review the `duplicates/` folder before deleting anything. The script never deletes files.
- Photos with suspicious dates land in `organized/unknown_date/` — you can manually sort those.

---

## What "Very Strict" Similarity Means

With `--hash-threshold 4` (default), two photos are flagged as duplicates only if they are nearly pixel-identical. This catches:

- ✅ Same photo saved twice (different filename)
- ✅ Same photo with slight JPEG re-compression
- ✅ Same photo with minor brightness/contrast tweak
- ❌ Same scene photographed twice (different shots)
- ❌ Cropped version of a photo (would need threshold ~8–10)

If you want to also catch cropped versions, re-run with `--hash-threshold 8`.
