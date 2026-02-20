# photo-organizer

# 📷 Photo Organizer & Deduplicator

Scans tens of thousands of photos, removes near-duplicates, and organizes everything into clean `YYYY/MM/` folders — all without touching your originals.

---

## What It Does

1. **Scans** your source folder recursively for all image files
2. **Extracts the most reliable date** from each photo using this priority:
   - EXIF `DateTimeOriginal` ← most trustworthy (set by camera at capture)
   - EXIF `DateTimeDigitized`
   - EXIF `DateTime` (can be modified by editors — used as fallback)
   - Filename pattern (e.g. `IMG_20181215_134500.jpg`)
   - File system date ← least trustworthy
3. **Flags suspicious dates** — years outside 1990–present are noted in the report
4. **Detects duplicates** using perceptual hashing (very strict: catches re-saves, crops, minor edits)
5. **Keeps the oldest copy** when duplicates are found
6. **Copies** organized photos to `output/organized/YYYY/MM/`
7. **Copies duplicates** to `output/duplicates/` for your review
8. **Generates an HTML report** with all actions, suspicious dates, and duplicate groups

> ✅ Your originals are **never moved or deleted**. Everything is copied.

---

## Requirements

### Python
You need Python 3.10 or newer.  
Download from: https://www.python.org/downloads/

### Install Dependencies

Open Command Prompt and run:

```
pip install Pillow piexif imagehash tqdm colorama
```

**For iPhone HEIC photos (optional but recommended):**
```
pip install pillow-heif
```

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
└── photo_organizer_report.html
```

---

## Understanding the Report

The HTML report (open in any browser) contains:

- **Summary stats** — totals, time taken
- **Suspicious dates** — photos where the date might be wrong (out-of-range years, no EXIF data)
- **Duplicate groups** — which file was kept and which were moved to `/duplicates`
- **Errors** — any files that couldn't be processed

---

## Supported Formats

JPG, JPEG, PNG, GIF, BMP, TIFF, WebP, HEIC/HEIF (with pillow-heif), and RAW formats: CR2, CR3, NEF, ARW, ORF, RW2, DNG, RAF, PEF, SRW

---

## Tips for 10,000+ Photos

- **Run `--dry-run` first** — always. The report will show you suspicious dates before anything moves.
- **Use `--workers 8`** on a modern machine to speed up the hashing phase.
- The hashing phase (visual similarity) is the slowest step — expect 1–3 hours for 50k photos on a typical machine.
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
