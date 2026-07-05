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
6. **Keeps the oldest copy** when duplicates are found
7. **Copies** organized photos and videos to `output/organized/YYYY/MM/`, together by date
8. **Copies duplicates** to `output/duplicates/` for your review
9. **Generates an HTML report** with all actions, suspicious dates, and duplicate groups

> ✅ Your originals are **never moved or deleted**. Everything is copied.

---

## Requirements

### Python
You need Python 3.10 or newer.  
Download from: https://www.python.org/downloads/

### Install Dependencies

Open Command Prompt and run:

```
pip install -r requirements.txt
```

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

---

## Output Structure

```
C:\Photos\Organized\
├── organized\
│   ├── 2018\
│   │   ├── 06\
│   │   │   ├── IMG_1234.jpg
│   │   │   └── DSC_0042.jpg
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
downloaded to this machine) are **left untouched in `--source`** — not copied to
`errors/`, since there's nothing to read yet. They're listed separately in the
report and in that run's `retry_photos_<timestamp>.txt`.

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

```
pip install -r requirements-dev.txt
pytest tests/
```

---

## Supported Formats

**Photos** (dated, deduplicated): JPG, JPEG, PNG, GIF, BMP, TIFF, WebP, HEIC/HEIF (with pillow-heif), and RAW formats: CR2, CR3, NEF, ARW, ORF, RW2, DNG, RAF, PEF, SRW

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
