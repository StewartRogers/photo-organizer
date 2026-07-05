"""
Photo Organizer & Deduplicator
================================
- Scans a source folder recursively for images and videos
- Reads EXIF DateTimeOriginal (most trustworthy) with fallbacks
- Detects visually similar images using perceptual hashing (very strict mode)
- Keeps the copy with the oldest EXIF date when duplicates are found
- Copies organized photos to a new folder structure: YYYY/MM/
- Videos are dated from container metadata (creation time — most trustworthy)
  with filename/filesystem fallbacks, and copied into the same YYYY/MM/ folder
  as photos, but are never deduplicated
- Moves suspected duplicates to a separate review folder
- Generates a detailed HTML report of all actions

Requirements (install with pip):
    pip install Pillow pillow-heif piexif imagehash tqdm colorama hachoir
    pip install pillow-avif-plugin  # optional, for AVIF support

Usage:
    python photo_organizer.py --source "C:\\Photos\\My Pictures" --output "C:\\Photos\\Organized"

Optional flags:
    --dry-run          Preview actions without copying anything
    --hash-threshold   Hamming distance for similarity (default: 4, lower = stricter)
    --workers          Parallel workers for hashing (default: 4)
    --log-file         Path to log file (default: <output>/photo_organizer.log)
    --retry-file       Reprocess only the photos listed in this file (see
                       <output>/retry_photos.txt from a previous run)
    --max-cloud-only-gb  Abort if more than this many GB are still cloud-only
                       (default: 1.0)
    --archive-source   After a fully successful run, zip --source in chunks
                       (written to --source's own root; see --archive-chunk-gb)
"""

import os
import sys
import shutil
import hashlib
import argparse
import logging
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import time

# Console output uses non-ASCII characters (⚠, ✓, ❌, emoji, ...). On a
# non-UTF-8 console (e.g. legacy Windows cp1252) this would otherwise raise
# UnicodeEncodeError and crash the run; reconfigure to UTF-8, replacing any
# character an unusual terminal still can't render instead of crashing.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

# Try to enable video container metadata (creation date) support
try:
    import hachoir.core.config as _hachoir_config
    _hachoir_config.quiet = True  # suppress hachoir's own "[warn] ..." console spam
    from hachoir.parser import createParser
    from hachoir.metadata import extractMetadata
    VIDEO_METADATA_SUPPORTED = True
except ImportError:
    VIDEO_METADATA_SUPPORTED = False

# ── Configuration ─────────────────────────────────────────────────────────────

SUPPORTED_PHOTO_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.heic', '.heif', '.cr2', '.cr3', '.nef', '.arw',
    '.orf', '.rw2', '.dng', '.raf', '.pef', '.srw', '.raw'
}

# Videos are organized by date alongside photos (so a trip's photos and clips
# land in the same YYYY/MM folder), but never deduplicated — no MD5/perceptual
# hashing is done for these, see process_photo().
SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mpg', '.mpeg', '.m4v', '.wmv',
    '.3gp', '.mkv', '.flv', '.webm'
}


def is_video(path: Path) -> bool:
    """True if `path`'s extension is a supported video format."""
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

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
REPORT_MAX_CLOUD_ONLY_ROWS = 200

# Windows-only: cloud-sync clients (OneDrive, etc.) mark a not-yet-downloaded
# "placeholder" file with this attribute. stat() succeeds on these (size/dates
# are available), but reading their content fails until they're hydrated.
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000

# ── Logging ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    """
    Log errors/warnings to a file, and route Python warnings (e.g. Pillow's
    palette/transparency UserWarning) there too instead of stderr, so they
    don't clutter stdout or interrupt the tqdm progress bars.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
    logging.captureWarnings(True)

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
    cloud_only: bool = False


@dataclass
class OrganizeResults:
    total: int
    organized: int
    duplicates: int
    errors: list = field(default_factory=list)
    errors_copied: int = 0
    cloud_only: list = field(default_factory=list)
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


def get_video_date(path: Path) -> tuple[Optional[datetime], str]:
    """
    Returns (datetime, source_description) using this priority:
      1. Container metadata (creation_time)  ← most trustworthy
      2. Filename pattern
      3. File system date                    ← least trustworthy

    Mirrors get_photo_date()'s fallback structure, but reads a video
    container's own creation-date metadata (MP4/MOV/AVI/MKV/... via hachoir)
    instead of EXIF, since Pillow can't open video files at all.
    """
    dt = None
    source = "unknown"

    # ── Container metadata ─────────────────────────────
    if VIDEO_METADATA_SUPPORTED:
        try:
            parser = createParser(str(path))
            if parser:
                try:
                    metadata = extractMetadata(parser)
                finally:
                    parser.stream.close()
                candidate = metadata.get("creation_date", None) if metadata else None
                if isinstance(candidate, datetime):
                    if not is_date_suspicious(candidate):
                        return candidate, "video metadata"
                    dt, source = candidate, "video metadata"
        except Exception:
            # Be conservative: don't let metadata parsing crash the run
            pass

    # ── Filename ──────────────────────────────────────
    fn_date = extract_date_from_filename(path)
    if fn_date and not is_date_suspicious(fn_date):
        return fn_date, "filename"

    # ── File system fallback ──────────────────────────
    try:
        stat = path.stat()
        fs_time = min(stat.st_mtime, stat.st_ctime)
        fs_date = datetime.fromtimestamp(fs_time)
        if not is_date_suspicious(fs_date):
            if dt:  # we had a suspicious video-metadata date — prefer filesystem
                return fs_date, "filesystem (video metadata suspicious)"
            return fs_date, "filesystem"
    except Exception:
        pass

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
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            img = img.convert("RGB")
            return imagehash.phash(img, hash_size=16)  # larger = more precise
    except Exception:
        return None


def safe_copy(src: Path, candidate: Path) -> tuple[Optional[Path], Optional[str]]:
    """
    Safely copy `src` to `candidate` without overwriting existing files.
    This function attempts to create the destination file using O_EXCL to
    guarantee we never overwrite an existing file. If the candidate exists,
    it will append _1, _2, ... to the stem until an unused name is found.
    Returns (path, None) on success, or (None, error_message) on failure.
    """
    dest_dir = candidate.parent
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return None, str(e)

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
        except OSError as e:
            return None, str(e)
        try:
            with os.fdopen(fd, 'wb') as out_f, open(src, 'rb') as in_f:
                shutil.copyfileobj(in_f, out_f, length=FILE_READ_CHUNK_SIZE)
            # Try to copy metadata; failures here are non-fatal
            try:
                shutil.copystat(str(src), str(target))
            except Exception:
                pass
            return target, None
        except Exception as e:
            try:
                target.unlink()
            except Exception:
                pass
            return None, str(e)


def is_cloud_only_placeholder(stat_result: os.stat_result) -> bool:
    """
    True if `stat_result` is a cloud-sync placeholder (e.g. a OneDrive Files
    On-Demand file) that hasn't been downloaded to this machine yet. Always
    False on platforms/objects without st_file_attributes (non-Windows).
    """
    attrs = getattr(stat_result, "st_file_attributes", 0)
    return bool(attrs & FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)


def hydrate_cloud_only_file(path: Path) -> bool:
    """
    Force a cloud-sync placeholder's content to download by reading it (e.g.
    OneDrive Files On-Demand hydrates on any real read, once its cloud file
    provider is healthy — verified live during the 2026-07-04 OneDrive
    incident, see CLAUDE.md). Returns True if the read succeeded (the file is
    now fully local), False otherwise (provider unavailable, etc.) — callers
    should fall back to treating the file as still cloud-only on False.
    """
    try:
        with open(path, "rb") as f:
            while f.read(FILE_READ_CHUNK_SIZE):
                pass
        return True
    except Exception:
        return False


def process_photo(path: Path, hydrate: bool = False) -> PhotoRecord:
    """
    Extract all metadata for one photo. Designed to run in a thread pool.

    `hydrate`: if True, actively download a cloud-only file (via
    hydrate_cloud_only_file()) and fold it into normal processing instead of
    skipping it — used when check_cloud_only_backlog() judged the backlog
    small enough to be worth downloading inline rather than deferring to a
    future --retry-file run.
    """
    record = PhotoRecord(path=str(path))
    try:
        stat = path.stat()
        record.size = stat.st_size

        if is_cloud_only_placeholder(stat):
            if not (hydrate and hydrate_cloud_only_file(path)):
                record.cloud_only = True
                return record
            # Hydration succeeded — the file is now fully local; fall
            # through and process it exactly like any other photo/video.

        video = is_video(path)
        dt, source = get_video_date(path) if video else get_photo_date(path)
        record.date = dt
        record.date_source = source

        # Videos are organized by date like photos, but never deduplicated:
        # leaving file_hash/phash unset means find_duplicates() (which only
        # groups records with a truthy hash) simply never considers them.
        if not video:
            record.file_hash = compute_file_hash(path)
            ph = compute_perceptual_hash(path)
            record.phash = str(ph) if ph else None

    except Exception as e:
        record.error = str(e)
        logger.error("Failed to process %s: %s", path, e)

    return record

# ── Duplicate Detection ────────────────────────────────────────────────────────

class _BKNode:
    """One entry in a BKTree, plus its children indexed by distance from it."""
    __slots__ = ("path", "hash_obj", "children")

    def __init__(self, path: str, hash_obj):
        self.path = path
        self.hash_obj = hash_obj
        self.children: dict[int, "_BKNode"] = {}


class BKTree:
    """
    BK-tree over perceptual hashes, supporting fast "all points within radius
    r of a query" lookups under Hamming distance. Used so find_duplicates()'s
    visual-similarity pass doesn't have to compare every photo against every
    other photo (which is O(n^2) and becomes hours-long above ~10-20k photos).

    Insertion and lookup rely only on the triangle inequality holding for the
    distance function (true for Hamming distance / imagehash's `-` operator),
    so this doesn't change which photos are found within `hash_threshold` of
    each other — only how quickly they're found.
    """

    def __init__(self):
        self.root: Optional[_BKNode] = None

    def add(self, path: str, hash_obj) -> None:
        if self.root is None:
            self.root = _BKNode(path, hash_obj)
            return
        node = self.root
        while True:
            d = abs(hash_obj - node.hash_obj)
            child = node.children.get(d)
            if child is None:
                node.children[d] = _BKNode(path, hash_obj)
                return
            node = child

    def query(self, hash_obj, radius: int) -> list[tuple[str, int]]:
        """Return [(path, distance), ...] for every entry within `radius` of hash_obj."""
        if self.root is None:
            return []
        results = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = abs(hash_obj - node.hash_obj)
            if d <= radius:
                results.append((node.path, d))
            lo, hi = d - radius, d + radius
            for child_dist, child in node.children.items():
                if lo <= child_dist <= hi:
                    stack.append(child)
        return results


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
              f"a very large --hash-threshold on a very large library can still be slow "
              f"in the worst case. Proceeding anyway.")

    # Greedy clustering: each not-yet-claimed photo becomes a group leader and
    # claims every not-yet-claimed photo within hash_threshold of it. A BKTree
    # is used to find those matches without comparing against every other
    # photo (see BKTree docstring) — this changes performance, not which
    # groups come out the other end.
    tree = BKTree()
    for path, hash_obj in phash_entries:
        tree.add(path, hash_obj)

    visited = set()
    visual_groups = []

    print(f"  Comparing {len(phash_entries):,} perceptual hashes (threshold={hash_threshold})...")

    for path_i, hash_i in tqdm(phash_entries, desc="  Visual compare", unit="photo"):
        if path_i in visited:
            continue
        group = [path_i]
        visited.add(path_i)
        for path_j, _dist in tree.query(hash_i, hash_threshold):
            if path_j == path_i or path_j in visited:
                continue
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
    Recursively find supported image and video files under `source`.

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
            suffix = p.suffix.lower()
            if suffix in SUPPORTED_PHOTO_EXTENSIONS or suffix in SUPPORTED_VIDEO_EXTENSIONS:
                found.append(p)
    return found


def load_retry_paths(retry_file: Path) -> list[Path]:
    """
    Load photo paths from a previous run's retry file (one path per line, as
    written by write_retry_file()), letting a follow-up run reprocess just the
    photos that didn't make it into organized/ or duplicates/ last time,
    instead of a full re-scan of --source.
    """
    lines = retry_file.read_text(encoding="utf-8").splitlines()
    return [Path(line) for line in lines if line.strip()]


def estimate_required_bytes(paths: list[Path]) -> tuple[int, int]:
    """
    Sum up file sizes for a worst-case destination-space estimate.

    organize_photos() never skips a file for being a duplicate — it copies it
    to duplicates/ instead of organized/ — so total destination usage is
    essentially the total size of every currently-readable photo, regardless
    of how many turn out to be duplicates. stat() reports the correct size
    for cloud-only placeholders without downloading them, so those bytes are
    counted separately (skipped this run, but may be needed for a future
    --retry-file run once they're synced).

    Returns (needed_bytes, cloud_only_bytes).
    """
    needed_bytes = 0
    cloud_only_bytes = 0
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        if is_cloud_only_placeholder(st):
            cloud_only_bytes += st.st_size
        else:
            needed_bytes += st.st_size
    return needed_bytes, cloud_only_bytes


def check_disk_space(paths: list[Path], output: Path, dry_run: bool) -> None:
    """
    Pre-flight worst-case space check, run before the (potentially
    hours-long) processing pass — so a full destination drive is caught
    immediately instead of partway through copying (see CLAUDE.md's
    2026-07-04 OneDrive incident, where this was discovered the hard way).
    Exits with an error if there isn't enough room, unless --dry-run (which
    never copies anything, so it's safe to preview regardless).
    """
    needed_bytes, cloud_only_bytes = estimate_required_bytes(paths)

    check_dir = output
    while not check_dir.exists():
        check_dir = check_dir.parent
    free_bytes = shutil.disk_usage(check_dir).free

    gb = 1024 ** 3
    print(f"{Fore.CYAN}Estimated space needed: {needed_bytes / gb:.2f} GB "
          f"(worst case — duplicates are copied too, not skipped)")
    if cloud_only_bytes:
        print(f"  + {cloud_only_bytes / gb:.2f} GB currently cloud-only "
              f"(not counted; skipped this run)")
    print(f"  Free on destination:    {free_bytes / gb:.2f} GB")

    if needed_bytes > free_bytes:
        shortfall_gb = (needed_bytes - free_bytes) / gb
        if dry_run:
            print(f"{Fore.YELLOW}⚠  Not enough free space for a real run: "
                  f"~{shortfall_gb:.2f} GB short. Continuing since this is --dry-run.")
        else:
            print(f"{Fore.RED}✗ Not enough free space: ~{shortfall_gb:.2f} GB short. "
                  f"Free up space or point --output at a different drive, then retry.")
            sys.exit(1)


def check_cloud_only_backlog(paths: list[Path], max_cloud_only_gb: float, dry_run: bool) -> bool:
    """
    Pre-flight check: if a large amount of source content is still cloud-only
    (e.g. OneDrive still downloading), a run right now would mostly just
    populate the retry list rather than actually organize anything. Exits and
    asks the user to try again once more has synced, unless --dry-run (warns
    only, since a dry run never copies anything anyway).

    Returns True if the backlog is small enough that cloud-only files
    encountered during processing should be actively downloaded and folded
    into this run (see hydrate_cloud_only_file()), rather than skipped for a
    future --retry-file run. Always False under --dry-run — a dry run must
    never touch the filesystem, and triggering a cloud download counts.
    """
    _, cloud_only_bytes = estimate_required_bytes(paths)
    max_cloud_only_bytes = max_cloud_only_gb * 1024 ** 3
    gb = 1024 ** 3

    if cloud_only_bytes <= max_cloud_only_bytes:
        if cloud_only_bytes and not dry_run:
            print(f"{Fore.CYAN}{cloud_only_bytes / gb:.2f} GB is cloud-only but under the "
                  f"{max_cloud_only_gb:.2f} GB threshold — will download these as they're "
                  f"encountered instead of skipping them.")
        return cloud_only_bytes > 0 and not dry_run

    message = (f"{cloud_only_bytes / gb:.2f} GB of source files are still cloud-only "
               f"(not downloaded yet) — more than the {max_cloud_only_gb:.2f} GB threshold.")
    if dry_run:
        print(f"{Fore.YELLOW}⚠  {message} Continuing since this is --dry-run.")
        return False
    else:
        print(f"{Fore.RED}✗ {message}")
        print(f"{Fore.RED}  Let OneDrive (or your cloud-sync provider) finish downloading more, "
              f"then try again — or raise --max-cloud-only-gb if this is expected.")
        sys.exit(1)

# ── Processing pipeline ─────────────────────────────────────────────────────────

def process_all_photos(paths: list[Path], workers: int, hydrate: bool = False) -> list[PhotoRecord]:
    """Extract dates and hashes for every photo, in parallel. See process_photo()
    for what `hydrate` does."""
    photos = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_photo, p, hydrate): p for p in paths}
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
    cloud_only = [p.path for p in photos if p.cloud_only]
    # Cloud-only photos never reach get_photo_date() (date stays None), so
    # exclude them here — they're not "suspicious", just not synced yet, and
    # are already reported in their own category above.
    suspicious = [p for p in photos
                  if not p.cloud_only and ("SUSPICIOUS" in p.date_source or p.date is None)]

    print(f"\n{Fore.CYAN}{'[DRY RUN] ' if dry_run else ''}Organizing photos...")

    dup_root = output / "duplicates"
    error_root = output / "errors"
    errors_copied = 0

    for photo in tqdm(photos, desc="  Copying", unit="photo"):
        path = Path(photo.path)

        if photo.cloud_only:
            # Not downloaded locally yet — nothing to copy. Leave it in
            # --source untouched; it's reported separately so it can be
            # retried (via --retry-file) once it's synced.
            continue

        if photo.error:
            if not dry_run:
                copied, copy_err = safe_copy(path, error_root / path.name)
                if copied:
                    errors_copied += 1
                else:
                    logger.error("Could not copy errored photo %s to errors/: %s",
                                 photo.path, copy_err)
            continue

        is_dup = photo.path in dup_paths
        candidate = (dup_root / path.name) if is_dup else destination_path(photo, output)

        if dry_run:
            if not is_dup:
                organized_count += 1
            continue

        copied, copy_err = safe_copy(path, candidate)
        if copied:
            if not is_dup:
                organized_count += 1
        else:
            kind = "duplicate" if is_dup else "organized photo"
            errors.append({"path": photo.path, "error": f"Failed to copy {kind}: {copy_err}"})
            logger.error("Failed to copy %s: %s (%s)", kind, photo.path, copy_err)

    return OrganizeResults(
        total=len(photos),
        organized=organized_count,
        duplicates=len(dup_paths),
        errors=errors,
        errors_copied=errors_copied,
        cloud_only=cloud_only,
        suspicious_dates=suspicious,
        dup_groups=dup_groups,
    )

# ── Report Generation ──────────────────────────────────────────────────────────

def build_report_html(results: OrganizeResults, output_root: Path, dry_run: bool, elapsed: float,
                       retry_path: Optional[Path] = None) -> str:
    """Render the HTML report from already-computed results."""
    total = results.total
    organized = results.organized
    dup_count = results.duplicates
    errors = results.errors
    cloud_only = results.cloud_only
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
  <div class="stat"><div class="num">{results.errors_copied:,}</div><div class="label">Copied to errors/</div></div>
  <div class="stat"><div class="num">{len(cloud_only):,}</div><div class="label">Cloud-only (not downloaded)</div></div>
  <div class="stat"><div class="num">{elapsed:.0f}s</div><div class="label">Processing time</div></div>
</div>

<h2>📁 Output Structure</h2>
<p>Organized photos → <code>{output_root / "organized"}</code><br>
Duplicates → <code>{output_root / "duplicates"}</code><br>
Errored photos → <code>{output_root / "errors"}</code></p>
{f"<p>Photos needing a retry (errors + cloud-only) → <code>{retry_path}</code> — use <code>--retry-file \"{retry_path}\"</code> on your next run.</p>" if retry_path else ""}
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
        html += "<p>Photos that failed date/hash processing were still copied to <code>errors/</code> for inspection; photos that failed to copy to their destination were not.</p>"
        html += "<table><tr><th>File</th><th>Error</th></tr>"
        for item in errors[:REPORT_MAX_ERROR_ROWS]:
            html += f"<tr><td>{Path(item['path']).name}</td><td>{item.get('error', '')}</td></tr>"
        if len(errors) > REPORT_MAX_ERROR_ROWS:
            html += f"<tr><td colspan='2'><em>...and {len(errors) - REPORT_MAX_ERROR_ROWS} more</em></td></tr>"
        html += "</table>"

    # Cloud-only (not-yet-downloaded) files table
    if cloud_only:
        html += "<h2>☁️ Cloud-Only (Not Downloaded)</h2>"
        html += ("<p>These files are cloud-sync placeholders (e.g. OneDrive Files On-Demand) "
                 "that haven't been downloaded to this machine yet, so their content couldn't "
                 "be read. They were left untouched in the source folder. Sync them locally "
                 "(e.g. right-click → \"Always keep on this device\") and re-run — "
                 "<code>--retry-file</code> can reprocess just these.</p>")
        html += "<table><tr><th>File</th></tr>"
        for item in cloud_only[:REPORT_MAX_CLOUD_ONLY_ROWS]:
            html += f"<tr><td>{Path(item).name}</td></tr>"
        if len(cloud_only) > REPORT_MAX_CLOUD_ONLY_ROWS:
            html += f"<tr><td><em>...and {len(cloud_only) - REPORT_MAX_CLOUD_ONLY_ROWS} more</em></td></tr>"
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


def generate_report(results: OrganizeResults, output_root: Path, dry_run: bool, elapsed: float,
                     retry_path: Optional[Path] = None) -> Path:
    """Write the HTML report to disk, atomically."""
    report_path = output_root / "photo_organizer_report.html"
    html = build_report_html(results, output_root, dry_run, elapsed, retry_path)

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


def write_retry_file(results: OrganizeResults, output_root: Path, run_id: str) -> Optional[Path]:
    """
    Write the photos that didn't get organized this run (errors + cloud-only
    placeholders not yet downloaded) to <output>/retry_photos_<run_id>.txt, one
    path per line, so a follow-up run can reprocess just these via
    --retry-file instead of a full re-scan. Each run gets its own file (rather
    than overwriting one shared name) so you can keep a history across runs
    and choose which one to retry from. Returns None (writing nothing) if
    there's nothing to retry.
    """
    paths = [item["path"] for item in results.errors] + list(results.cloud_only)
    if not paths:
        return None
    retry_path = output_root / f"retry_photos_{run_id}.txt"
    retry_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return retry_path


ARCHIVE_EXTENSION = ".zip"


def _verify_zip(zip_path: Path) -> Optional[str]:
    """
    Return None if `zip_path` is a valid, uncorrupted zip archive, or a
    description of the problem otherwise. Uses zipfile's own CRC-checking
    testzip() (reads every member and verifies its checksum) rather than
    just confirming the file opens.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_name = zf.testzip()
            return f"corrupt member: {bad_name}" if bad_name else None
    except Exception as e:
        return str(e)


def archive_source(source: Path, chunk_gb: float, run_id: str) -> tuple[list[Path], list[tuple[str, str]]]:
    """
    Pack every file under `source` into a sequence of independent, fully
    valid zip archives (~chunk_gb each, greedily filled — a single file
    larger than chunk_gb still gets its own, larger chunk rather than being
    split), written directly into `source`'s root as
    archive_<run_id>_NNNN.zip. --source's existing files are never modified
    or removed — this only adds new files alongside them.

    Uses ZIP_STORED (no compression): photos/videos are already compressed
    formats, so deflate would mostly just spend CPU without shrinking
    output. Existing archive_*.zip files (e.g. from a previous run) are
    skipped when walking, so re-running this doesn't zip its own output.

    Each finished chunk is immediately verified with testzip() (see
    _verify_zip()); any corrupt or unreadable chunk is reported back rather
    than silently trusted.

    Returns (zip_paths, problems) where problems is a list of
    (file_or_archive_path, reason) for anything that couldn't be archived or
    failed verification.
    """
    chunk_bytes = chunk_gb * 1024 ** 3
    problems: list[tuple[str, str]] = []

    all_files = []
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or p.suffix.lower() == ARCHIVE_EXTENSION:
                continue
            all_files.append(p)

    if not all_files:
        return [], problems

    zip_paths: list[Path] = []
    zf: Optional[zipfile.ZipFile] = None
    current_size = 0
    chunk_index = 0

    def open_next_chunk():
        nonlocal zf, current_size, chunk_index
        close_current_chunk()
        chunk_index += 1
        chunk_path = source / f"archive_{run_id}_{chunk_index:04d}{ARCHIVE_EXTENSION}"
        zf = zipfile.ZipFile(chunk_path, "w", compression=zipfile.ZIP_STORED)
        zip_paths.append(chunk_path)
        current_size = 0

    def close_current_chunk():
        nonlocal zf
        if zf is None:
            return
        chunk_path = Path(zf.filename)
        zf.close()
        zf = None
        bad = _verify_zip(chunk_path)
        if bad:
            problems.append((str(chunk_path), bad))

    open_next_chunk()
    for p in tqdm(all_files, desc="  Archiving", unit="file"):
        try:
            size = p.stat().st_size
        except OSError as e:
            problems.append((str(p), str(e)))
            continue

        if current_size > 0 and current_size + size > chunk_bytes:
            open_next_chunk()

        try:
            arcname = str(p.relative_to(source))
        except ValueError:
            arcname = p.name

        try:
            zf.write(p, arcname=arcname)
            current_size += size
        except Exception as e:
            problems.append((str(p), str(e)))

    close_current_chunk()

    return zip_paths, problems

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photo Organizer & Deduplicator")
    parser.add_argument("--source", required=True, help="Source folder containing your photos")
    parser.add_argument("--output", required=True, help="Output root folder")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't copy files")
    parser.add_argument("--hash-threshold", type=int, default=4,
                        help="Perceptual hash distance threshold (default 4 = very strict; 0 = exact visual match only)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for hashing")
    parser.add_argument("--log-file", help="Path to log file (default: <output>/photo_organizer.log)")
    parser.add_argument("--retry-file",
                        help="Reprocess only the photos listed in this file (one path per line) "
                             "instead of rescanning --source; see <output>/retry_photos.txt "
                             "from a previous run")
    parser.add_argument("--max-cloud-only-gb", type=float, default=1.0,
                        help="Abort before processing if more than this many GB of source files "
                             "are still cloud-only / not downloaded yet (default: 1.0). Raise this "
                             "if running against a partially-synced library is expected.")
    parser.add_argument("--archive-source", action="store_true",
                        help="After a fully successful run (zero errors, zero cloud-only), pack "
                             "--source into ~--archive-chunk-gb zip files written to --source's "
                             "own root (archive_<run_id>_NNNN.zip). --source's existing files are "
                             "never modified; skipped under --dry-run or if the run wasn't clean.")
    parser.add_argument("--archive-chunk-gb", type=float, default=1.0,
                        help="Target size per zip chunk when using --archive-source (default: 1.0)")
    return parser.parse_args()


def resolve_log_path(log_file_arg: Optional[str], output: Path) -> Path:
    """Resolve the --log-file argument to a concrete path, defaulting to
    <output>/photo_organizer.log."""
    return Path(log_file_arg) if log_file_arg else output / "photo_organizer.log"


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


def print_summary(results: OrganizeResults, report_path: Path, log_path: Path,
                   retry_path: Optional[Path], elapsed: float, dry_run: bool) -> None:
    print(f"\n{Fore.GREEN}{'─' * 50}")
    print(f"{Fore.GREEN}✓ Done in {elapsed:.1f}s")
    print(f"  Total scanned:      {results.total:,}")
    print(f"  Organized:          {results.organized:,}")
    print(f"  Duplicates found:   {results.duplicates:,}")
    print(f"  Suspicious dates:   {len(results.suspicious_dates):,}")
    print(f"  Errors:             {len(results.errors):,}")
    print(f"  Copied to errors/:  {results.errors_copied:,}")
    print(f"  Cloud-only:         {len(results.cloud_only):,}")
    print(f"\n  📄 Report: {report_path}")
    print(f"  📝 Log:    {log_path}")
    if retry_path:
        print(f"  🔁 Retry:  {retry_path}  (rerun with --retry-file \"{retry_path}\")")
    if dry_run:
        print(f"\n{Fore.YELLOW}  ⚠  This was a DRY RUN. Re-run without --dry-run to copy files.")


def main():
    args = parse_args()
    source, output = validate_paths(Path(args.source), Path(args.output))

    log_path = resolve_log_path(args.log_file, output)
    setup_logging(log_path)

    if not HEIC_SUPPORTED:
        print(f"{Fore.YELLOW}⚠  pillow-heif not installed — HEIC files will be skipped.")
        print(f"   Install with: pip install pillow-heif\n")

    if not VIDEO_METADATA_SUPPORTED:
        print(f"{Fore.YELLOW}⚠  hachoir not installed — video dates will use "
              f"filename/filesystem only (no container metadata).")
        print(f"   Install with: pip install hachoir\n")

    start_time = time.time()
    run_id = datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M%S")

    # ── 1. Scan ────────────────────────────────────────────────────────────────
    if args.retry_file:
        retry_file_path = Path(args.retry_file)
        print(f"{Fore.CYAN}Loading retry list from {retry_file_path}...")
        all_paths = load_retry_paths(retry_file_path)
        print(f"  Loaded {len(all_paths):,} photos to retry")
    else:
        print(f"{Fore.CYAN}Scanning {source} for photos...")
        all_paths = scan_photos(source)
        print(f"  Found {len(all_paths):,} image files")

    if not all_paths:
        print(f"{Fore.YELLOW}No images found. Check your source path and supported extensions.")
        sys.exit(0)

    check_disk_space(all_paths, output, args.dry_run)
    hydrate = check_cloud_only_backlog(all_paths, args.max_cloud_only_gb, args.dry_run)

    # ── 2. Process (parallel) ─────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}Extracting dates & computing hashes ({args.workers} workers)...")
    photos = process_all_photos(all_paths, args.workers, hydrate)

    # ── 3. Find duplicates ────────────────────────────────────────────────────
    dup_groups = find_duplicates(photos, args.hash_threshold)

    # ── 4. Copy / organize ────────────────────────────────────────────────────
    results = organize_photos(photos, dup_groups, output, args.dry_run)

    elapsed = time.time() - start_time

    # ── 5. Report ─────────────────────────────────────────────────────────────
    retry_path = write_retry_file(results, output, run_id)
    report_path = generate_report(results, output, args.dry_run, elapsed, retry_path)

    print_summary(results, report_path, log_path, retry_path, elapsed, args.dry_run)

    # ── 6. Optional: archive source ───────────────────────────────────────────
    if args.archive_source:
        if args.dry_run:
            print(f"{Fore.YELLOW}⚠  Skipping --archive-source: dry runs never create real output.")
        elif results.errors or results.cloud_only:
            print(f"{Fore.YELLOW}⚠  Skipping --archive-source: this run wasn't fully clean "
                  f"({len(results.errors):,} errors, {len(results.cloud_only):,} cloud-only) — "
                  f"resolve those and re-run first.")
        else:
            print(f"\n{Fore.CYAN}Archiving --source into ~{args.archive_chunk_gb:.2f} GB zip "
                  f"chunks (written to {source})...")
            zip_paths, problems = archive_source(source, args.archive_chunk_gb, run_id)
            print(f"{Fore.GREEN}✓ Created {len(zip_paths):,} verified zip file(s) in {source}")
            if problems:
                print(f"{Fore.RED}✗ {len(problems):,} problem(s) during archiving (see log):")
                for path_str, reason in problems:
                    logger.error("Archive problem for %s: %s", path_str, reason)
                    print(f"    {path_str}: {reason}")


if __name__ == "__main__":
    main()
