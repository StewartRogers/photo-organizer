"""
Photo Organizer & Deduplicator
================================
- Scans a source folder recursively for images
- Reads EXIF DateTimeOriginal (most trustworthy) with fallbacks
- Detects visually similar images using perceptual hashing (very strict mode)
- Keeps the copy with the oldest EXIF date when duplicates are found
- Copies organized photos to a new folder structure: YYYY/MM/
- Moves suspected duplicates to a separate review folder
- Generates a detailed HTML report of all actions

Requirements (install with pip):
    pip install Pillow pillow-heif piexif imagehash tqdm colorama
    pip install pillow-avif-plugin  # optional, for AVIF support

Usage:
    python photo_organizer.py --source "C:\\Photos\\My Pictures" --output "C:\\Photos\\Organized"

Optional flags:
    --dry-run          Preview actions without copying anything
    --hash-threshold   Hamming distance for similarity (default: 4, lower = stricter)
    --workers          Parallel workers for hashing (default: 4)
"""

import os
import sys
import shutil
import hashlib
import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import time

try:
    from PIL import Image, UnidentifiedImageError
    import piexif
    import imagehash
    from tqdm import tqdm
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install requirements:")
    print("  pip install Pillow piexif imagehash tqdm colorama")
    sys.exit(1)

# Try to register HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# ── Configuration ─────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.heic', '.heif', '.cr2', '.cr3', '.nef', '.arw',
    '.orf', '.rw2', '.dng', '.raf', '.pef', '.srw', '.raw'
}

# Suspicious year range — flag dates outside this as potentially wrong
MIN_VALID_YEAR = 1990
MAX_VALID_YEAR = datetime.now().year + 1

FILE_READ_CHUNK_SIZE = 65536  # bytes per chunk when hashing/copying files

# Perceptual-hash clustering in find_duplicates() is O(n^2). Above this many
# candidates it can take a very long time; we warn (but still proceed) so
# large libraries don't appear to hang silently.
VISUAL_DEDUP_WARN_THRESHOLD = 50_000

REPORT_MAX_SUSPICIOUS_ROWS = 500
REPORT_MAX_ERROR_ROWS = 200
REPORT_MAX_DUP_GROUP_ROWS = 100

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class PhotoRecord:
    """All metadata collected for a single photo."""
    path: str
    size: int = 0
    date: Optional[datetime] = None
    date_source: str = "error"
    file_hash: Optional[str] = None
    phash: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OrganizeResults:
    total: int
    organized: int
    duplicates: int
    errors: list = field(default_factory=list)
    suspicious_dates: list = field(default_factory=list)
    dup_groups: dict = field(default_factory=dict)

# ── Date Extraction ────────────────────────────────────────────────────────────

def parse_exif_date(date_str: str) -> Optional[datetime]:
    """Parse EXIF date string 'YYYY:MM:DD HH:MM:SS'"""
    if not date_str:
        return None
    try:
        # EXIF standard format
        return datetime.strptime(date_str.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        pass
    # Try some non-standard formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_date_from_filename(path: Path) -> Optional[datetime]:
    """Try to pull a date out of the filename itself."""
    name = path.stem
    # Common patterns: IMG_20181215, 2018-12-15, 20181215_134500, etc.
    patterns = [
        r'(\d{4})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})',  # full datetime
        r'(\d{4})[_\-](\d{2})[_\-](\d{2})',   # date only with separators
        r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',  # compact datetime
        r'(\d{4})(\d{2})(\d{2})',              # compact date
    ]
    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            groups = m.groups()
            try:
                if len(groups) == 6:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                    int(groups[3]), int(groups[4]), int(groups[5]))
                elif len(groups) == 3:
                    y, mo, d = int(groups[0]), int(groups[1]), int(groups[2])
                    if MIN_VALID_YEAR <= y <= MAX_VALID_YEAR and 1 <= mo <= 12 and 1 <= d <= 31:
                        return datetime(y, mo, d)
            except ValueError:
                continue
    return None


def is_date_suspicious(dt: Optional[datetime]) -> bool:
    """Return True if the date looks wrong."""
    if dt is None:
        return True
    return not (MIN_VALID_YEAR <= dt.year <= MAX_VALID_YEAR)


def get_photo_date(path: Path) -> tuple[Optional[datetime], str]:
    """
    Returns (datetime, source_description) using this priority:
      1. EXIF DateTimeOriginal  ← most trustworthy
      2. EXIF DateTimeDigitized
      3. EXIF DateTime
      4. Filename pattern
      5. File creation date     ← least trustworthy
    """
    dt = None
    source = "unknown"

    # ── EXIF ──────────────────────────────────────────
    try:
        with Image.open(path) as img:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                try:
                    exif = piexif.load(exif_bytes)
                except Exception:
                    exif = {}
                exif_map = [
                    (piexif.ExifIFD.DateTimeOriginal,  exif.get("Exif", {}), "EXIF:DateTimeOriginal"),
                    (piexif.ExifIFD.DateTimeDigitized, exif.get("Exif", {}), "EXIF:DateTimeDigitized"),
                    (piexif.ImageIFD.DateTime,          exif.get("0th",  {}), "EXIF:DateTime"),
                ]
                for tag, ifd, label in exif_map:
                    raw = ifd.get(tag)
                    if raw:
                        val = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
                        candidate = parse_exif_date(val)
                        if candidate and not is_date_suspicious(candidate):
                            return candidate, label
                        elif candidate and dt is None:
                            dt, source = candidate, label  # store even if suspicious
    except UnidentifiedImageError:
        # Not an image we can open — fall through to other strategies
        pass
    except Exception:
        # Be conservative: don't let EXIF parsing crash the run
        pass

    # ── Filename ──────────────────────────────────────
    fn_date = extract_date_from_filename(path)
    if fn_date and not is_date_suspicious(fn_date):
        return fn_date, "filename"

    # ── File system fallback ──────────────────────────
    try:
        stat = path.stat()
        # On Windows, st_ctime is creation time; on Unix it's metadata change time
        fs_time = min(stat.st_mtime, stat.st_ctime)
        fs_date = datetime.fromtimestamp(fs_time)
        if not is_date_suspicious(fs_date):
            if dt:  # we had a suspicious EXIF date — prefer filesystem
                return fs_date, "filesystem (EXIF date suspicious)"
            return fs_date, "filesystem"
    except Exception:
        pass

    # Return suspicious date if that's all we have
    if dt:
        return dt, f"{source} (SUSPICIOUS — year {dt.year})"

    return None, "no date found"

# ── Hashing & Deduplication ────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> str:
    """MD5 of file content for exact duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(FILE_READ_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_perceptual_hash(path: Path) -> Optional[imagehash.ImageHash]:
    """Perceptual hash for visual similarity."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img, hash_size=16)  # larger = more precise
    except Exception:
        return None


def safe_copy(src: Path, candidate: Path) -> Optional[Path]:
    """
    Safely copy `src` to `candidate` without overwriting existing files.
    This function attempts to create the destination file using O_EXCL to
    guarantee we never overwrite an existing file. If the candidate exists,
    it will append _1, _2, ... to the stem until an unused name is found.
    Returns the path of the copied file, or None on error.
    """
    dest_dir = candidate.parent
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 0
    while True:
        if counter == 0:
            target = dest_dir / f"{stem}{suffix}"
        else:
            target = dest_dir / f"{stem}_{counter}{suffix}"
        try:
            # Use low-level os.open with O_EXCL to ensure we don't overwrite
            fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            counter += 1
            continue
        except OSError:
            return None
        try:
            with os.fdopen(fd, 'wb') as out_f, open(src, 'rb') as in_f:
                shutil.copyfileobj(in_f, out_f, length=FILE_READ_CHUNK_SIZE)
            # Try to copy metadata; failures here are non-fatal
            try:
                shutil.copystat(str(src), str(target))
            except Exception:
                pass
            return target
        except Exception:
            try:
                target.unlink()
            except Exception:
                pass
            return None


def process_photo(path: Path) -> PhotoRecord:
    """Extract all metadata for one photo. Designed to run in a thread pool."""
    record = PhotoRecord(path=str(path))
    try:
        stat = path.stat()
        record.size = stat.st_size

        dt, source = get_photo_date(path)
        record.date = dt
        record.date_source = source

        record.file_hash = compute_file_hash(path)
        ph = compute_perceptual_hash(path)
        record.phash = str(ph) if ph else None

    except Exception as e:
        record.error = str(e)

    return record

# ── Duplicate Detection ────────────────────────────────────────────────────────

def find_duplicates(photos: list[PhotoRecord], hash_threshold: int) -> dict[str, list[str]]:
    """
    Groups photos into duplicate sets.
    Returns {canonical_path: [duplicate_path, ...]}

    Strategy:
      1. Exact match (same file hash) → definite duplicates
      2. Perceptual hash within threshold → visual duplicates

    Within each group, keep the photo with the OLDEST date.
    """
    print(f"\n{Fore.CYAN}Grouping duplicates...")

    # ── Step 1: exact hash groups ──────────────────────────────────────────────
    hash_groups = defaultdict(list)
    for p in photos:
        if p.file_hash:
            hash_groups[p.file_hash].append(p)

    exact_groups = {k: v for k, v in hash_groups.items() if len(v) > 1}

    # Build set of paths already assigned to an exact group
    exact_paths = set()
    for group in exact_groups.values():
        for p in group:
            exact_paths.add(p.path)

    # ── Step 2: perceptual hash clustering ────────────────────────────────────
    # Only run on photos NOT already in an exact group
    remaining = [p for p in photos if p.path not in exact_paths and p.phash]

    # Build a list of (path, hash_obj)
    phash_entries = []
    for p in remaining:
        try:
            phash_entries.append((p.path, imagehash.hex_to_hash(p.phash)))
        except Exception:
            pass

    if len(phash_entries) > VISUAL_DEDUP_WARN_THRESHOLD:
        print(f"{Fore.YELLOW}⚠  {len(phash_entries):,} photos need visual comparison — "
              f"this step is O(n²) and may take a long time above "
              f"{VISUAL_DEDUP_WARN_THRESHOLD:,}. Proceeding anyway.")

    # Greedy clustering: O(n²) but fine for <50k images with early exit
    visited = set()
    visual_groups = []

    print(f"  Comparing {len(phash_entries):,} perceptual hashes (threshold={hash_threshold})...")

    for i, (path_i, hash_i) in enumerate(tqdm(phash_entries, desc="  Visual compare", unit="photo")):
        if path_i in visited:
            continue
        group = [path_i]
        visited.add(path_i)
        for j in range(i + 1, len(phash_entries)):
            path_j, hash_j = phash_entries[j]
            if path_j in visited:
                continue
            if abs(hash_i - hash_j) <= hash_threshold:
                group.append(path_j)
                visited.add(path_j)
        if len(group) > 1:
            visual_groups.append(group)

    # ── Decide keeper for each group ──────────────────────────────────────────
    duplicates = {}  # keeper_path → [dup_path, ...]
    photo_index = {p.path: p for p in photos}

    def pick_keeper(paths: list[str]) -> str:
        """Pick the photo with the oldest valid date; fall back to largest file."""
        def sort_key(path):
            p = photo_index.get(path)
            dt = p.date if p and p.date else datetime.max
            size = p.size if p else 0
            return (dt, -size)
        return sorted(paths, key=sort_key)[0]

    for group in exact_groups.values():
        paths = [p.path for p in group]
        keeper = pick_keeper(paths)
        dups = [p for p in paths if p != keeper]
        duplicates[keeper] = duplicates.get(keeper, []) + dups

    for group in visual_groups:
        keeper = pick_keeper(group)
        dups = [p for p in group if p != keeper]
        duplicates[keeper] = duplicates.get(keeper, []) + dups

    return duplicates

# ── Output Structure ───────────────────────────────────────────────────────────

def destination_path(photo: PhotoRecord, output_root: Path) -> Path:
    """Compute the output path for an organized photo."""
    original = Path(photo.path)

    if photo.date:
        folder = output_root / "organized" / f"{photo.date.year:04d}" / f"{photo.date.month:02d}"
    else:
        folder = output_root / "organized" / "unknown_date"

    # Return base destination (unique name guaranteed by safe_copy at write time)
    return folder / original.name

# ── Scanning ────────────────────────────────────────────────────────────────────

def scan_photos(source: Path) -> list[Path]:
    """
    Recursively find supported image files under `source`.

    Symlinked files and directories are skipped: a symlink inside the source
    tree could point outside of it (e.g. onto an untrusted removable drive),
    which would otherwise let this tool read and copy arbitrary files that
    were never intended to be part of the photo library.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(p)
    return found

# ── Processing pipeline ─────────────────────────────────────────────────────────

def process_all_photos(paths: list[Path], workers: int) -> list[PhotoRecord]:
    """Extract dates and hashes for every photo, in parallel."""
    photos = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_photo, p): p for p in paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Processing", unit="photo"):
            photos.append(future.result())
    return photos


def organize_photos(photos: list[PhotoRecord], dup_groups: dict[str, list[str]],
                     output: Path, dry_run: bool) -> OrganizeResults:
    """Copy photos into organized/ and duplicates/ folders and collect results."""
    dup_paths = set()
    for dups in dup_groups.values():
        dup_paths.update(dups)

    organized_count = 0
    errors = [{"path": p.path, "error": p.error} for p in photos if p.error]
    suspicious = [p for p in photos if "SUSPICIOUS" in p.date_source or p.date is None]

    print(f"\n{Fore.CYAN}{'[DRY RUN] ' if dry_run else ''}Organizing photos...")

    dup_root = output / "duplicates"

    for photo in tqdm(photos, desc="  Copying", unit="photo"):
        if photo.error:
            continue

        path = Path(photo.path)
        is_dup = photo.path in dup_paths
        candidate = (dup_root / path.name) if is_dup else destination_path(photo, output)

        if dry_run:
            if not is_dup:
                organized_count += 1
            continue

        copied = safe_copy(path, candidate)
        if copied:
            if not is_dup:
                organized_count += 1
        else:
            kind = "duplicate" if is_dup else "organized photo"
            errors.append({"path": photo.path, "error": f"Failed to copy {kind}"})

    return OrganizeResults(
        total=len(photos),
        organized=organized_count,
        duplicates=len(dup_paths),
        errors=errors,
        suspicious_dates=suspicious,
        dup_groups=dup_groups,
    )

# ── Report Generation ──────────────────────────────────────────────────────────

def build_report_html(results: OrganizeResults, output_root: Path, dry_run: bool, elapsed: float) -> str:
    """Render the HTML report from already-computed results."""
    total = results.total
    organized = results.organized
    dup_count = results.duplicates
    errors = results.errors
    suspicious = results.suspicious_dates

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Photo Organizer Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; background: #f9f9f9; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 12px; }}
  h2 {{ color: #2c3e50; margin-top: 32px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 16px; margin: 24px 0; }}
  .stat {{ background: white; border-radius: 10px; padding: 20px; text-align: center;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .stat .num {{ font-size: 2em; font-weight: 700; color: #4a90d9; }}
  .stat .label {{ font-size: .85em; color: #666; margin-top: 4px; }}
  .dry-run-banner {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
                      padding: 12px 20px; margin-bottom: 24px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); border-radius: 8px; overflow: hidden; }}
  th {{ background: #4a90d9; color: white; padding: 10px 14px; text-align: left; font-size: .9em; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #eee; font-size: .85em; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f7ff; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
          font-size: .75em; font-weight: 600; margin: 1px; }}
  .tag-ok {{ background: #d4edda; color: #155724; }}
  .tag-warn {{ background: #fff3cd; color: #856404; }}
  .tag-err {{ background: #f8d7da; color: #721c24; }}
  .tag-dup {{ background: #d1ecf1; color: #0c5460; }}
  .footer {{ margin-top: 40px; color: #999; font-size: .8em; text-align: center; }}
</style>
</head>
<body>
<h1>📷 Photo Organizer Report</h1>
{"<div class='dry-run-banner'>⚠️ DRY RUN — No files were actually copied. Re-run without --dry-run to apply changes.</div>" if dry_run else ""}
<div class="stats">
  <div class="stat"><div class="num">{total:,}</div><div class="label">Photos scanned</div></div>
  <div class="stat"><div class="num">{organized:,}</div><div class="label">Photos organized</div></div>
  <div class="stat"><div class="num">{dup_count:,}</div><div class="label">Duplicates found</div></div>
  <div class="stat"><div class="num">{len(suspicious):,}</div><div class="label">Suspicious dates</div></div>
  <div class="stat"><div class="num">{len(errors):,}</div><div class="label">Errors</div></div>
  <div class="stat"><div class="num">{elapsed:.0f}s</div><div class="label">Processing time</div></div>
</div>

<h2>📁 Output Structure</h2>
<p>Organized photos → <code>{output_root / "organized"}</code><br>
Duplicates → <code>{output_root / "duplicates"}</code></p>
"""

    # Suspicious dates table
    if suspicious:
        html += "<h2>⚠️ Suspicious / Unverifiable Dates</h2>"
        html += "<p>These photos had missing, inconsistent, or out-of-range dates. They were organized using the best available date but should be reviewed.</p>"
        html += "<table><tr><th>File</th><th>Date Used</th><th>Source</th></tr>"
        for item in suspicious[:REPORT_MAX_SUSPICIOUS_ROWS]:
            date_str = item.date.isoformat() if item.date else "—"
            html += f"<tr><td>{Path(item.path).name}</td><td>{date_str}</td><td>{item.date_source or '—'}</td></tr>"
        if len(suspicious) > REPORT_MAX_SUSPICIOUS_ROWS:
            html += f"<tr><td colspan='3'><em>...and {len(suspicious) - REPORT_MAX_SUSPICIOUS_ROWS} more</em></td></tr>"
        html += "</table>"

    # Error table
    if errors:
        html += "<h2>❌ Errors</h2>"
        html += "<table><tr><th>File</th><th>Error</th></tr>"
        for item in errors[:REPORT_MAX_ERROR_ROWS]:
            html += f"<tr><td>{Path(item['path']).name}</td><td>{item.get('error', '')}</td></tr>"
        html += "</table>"

    # Duplicate groups
    if results.dup_groups:
        html += "<h2>🔁 Duplicate Groups (first 100)</h2>"
        html += "<table><tr><th>Kept</th><th>Duplicates moved</th></tr>"
        for keeper, dups in list(results.dup_groups.items())[:REPORT_MAX_DUP_GROUP_ROWS]:
            dup_names = "<br>".join(Path(d).name for d in dups)
            html += f"<tr><td>{Path(keeper).name}</td><td>{dup_names}</td></tr>"
        html += "</table>"

    html += f"<div class='footer'>Generated by photo_organizer.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
    html += "</body></html>"
    return html


def generate_report(results: OrganizeResults, output_root: Path, dry_run: bool, elapsed: float) -> Path:
    """Write the HTML report to disk, atomically."""
    report_path = output_root / "photo_organizer_report.html"
    html = build_report_html(results, output_root, dry_run, elapsed)

    output_root.mkdir(parents=True, exist_ok=True)
    # Write report atomically to avoid partial files
    tmp_path = output_root / (report_path.name + ".tmp")
    try:
        tmp_path.write_text(html, encoding="utf-8")
        os.replace(str(tmp_path), str(report_path))
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return report_path

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photo Organizer & Deduplicator")
    parser.add_argument("--source", required=True, help="Source folder containing your photos")
    parser.add_argument("--output", required=True, help="Output root folder")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't copy files")
    parser.add_argument("--hash-threshold", type=int, default=4,
                        help="Perceptual hash distance threshold (default 4 = very strict; 0 = exact visual match only)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for hashing")
    return parser.parse_args()


def validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    """
    Resolve --source/--output to absolute paths and reject configurations
    that would cause the tool to scan or write over itself:
      - source must exist
      - output must not be the same as, or nested inside, source
      - source must not be nested inside output
    """
    source = source.resolve()
    output = output.resolve()

    if not source.exists():
        print(f"{Fore.RED}Source folder not found: {source}")
        sys.exit(1)

    if output == source or output in source.parents or source in output.parents:
        print(f"{Fore.RED}--output cannot be the same as, or nested inside, --source (and vice versa).")
        print(f"  source: {source}")
        print(f"  output: {output}")
        sys.exit(1)

    return source, output


def print_summary(results: OrganizeResults, report_path: Path, elapsed: float, dry_run: bool) -> None:
    print(f"\n{Fore.GREEN}{'─' * 50}")
    print(f"{Fore.GREEN}✓ Done in {elapsed:.1f}s")
    print(f"  Total scanned:      {results.total:,}")
    print(f"  Organized:          {results.organized:,}")
    print(f"  Duplicates found:   {results.duplicates:,}")
    print(f"  Suspicious dates:   {len(results.suspicious_dates):,}")
    print(f"  Errors:             {len(results.errors):,}")
    print(f"\n  📄 Report: {report_path}")
    if dry_run:
        print(f"\n{Fore.YELLOW}  ⚠  This was a DRY RUN. Re-run without --dry-run to copy files.")


def main():
    args = parse_args()
    source, output = validate_paths(Path(args.source), Path(args.output))

    if not HEIC_SUPPORTED:
        print(f"{Fore.YELLOW}⚠  pillow-heif not installed — HEIC files will be skipped.")
        print(f"   Install with: pip install pillow-heif\n")

    start_time = time.time()

    # ── 1. Scan ────────────────────────────────────────────────────────────────
    print(f"{Fore.CYAN}Scanning {source} for photos...")
    all_paths = scan_photos(source)
    print(f"  Found {len(all_paths):,} image files")

    if not all_paths:
        print(f"{Fore.YELLOW}No images found. Check your source path and supported extensions.")
        sys.exit(0)

    # ── 2. Process (parallel) ─────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}Extracting dates & computing hashes ({args.workers} workers)...")
    photos = process_all_photos(all_paths, args.workers)

    # ── 3. Find duplicates ────────────────────────────────────────────────────
    dup_groups = find_duplicates(photos, args.hash_threshold)

    # ── 4. Copy / organize ────────────────────────────────────────────────────
    results = organize_photos(photos, dup_groups, output, args.dry_run)

    elapsed = time.time() - start_time

    # ── 5. Report ─────────────────────────────────────────────────────────────
    report_path = generate_report(results, output, args.dry_run, elapsed)

    print_summary(results, report_path, elapsed, args.dry_run)


if __name__ == "__main__":
    main()
