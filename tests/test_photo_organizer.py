import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import photo_organizer as po

import imagehash
from PIL import Image


# ── parse_exif_date ─────────────────────────────────────────────────────────

def test_parse_exif_date_standard_format():
    assert po.parse_exif_date("2020:05:17 14:30:00") == datetime(2020, 5, 17, 14, 30, 0)


def test_parse_exif_date_dash_format():
    assert po.parse_exif_date("2020-05-17 14:30:00") == datetime(2020, 5, 17, 14, 30, 0)


def test_parse_exif_date_date_only():
    assert po.parse_exif_date("2020:05:17") == datetime(2020, 5, 17)


def test_parse_exif_date_empty_or_none():
    assert po.parse_exif_date("") is None
    assert po.parse_exif_date(None) is None


def test_parse_exif_date_garbage():
    assert po.parse_exif_date("not a date") is None


# ── extract_date_from_filename ──────────────────────────────────────────────

def test_extract_date_from_filename_compact_datetime():
    dt = po.extract_date_from_filename(Path("IMG_20181215_134500.jpg"))
    assert dt == datetime(2018, 12, 15, 13, 45, 0)


def test_extract_date_from_filename_compact_date_only():
    dt = po.extract_date_from_filename(Path("IMG_20181215.jpg"))
    assert dt == datetime(2018, 12, 15)


def test_extract_date_from_filename_dashed_date():
    dt = po.extract_date_from_filename(Path("vacation-2019-07-04.png"))
    assert dt == datetime(2019, 7, 4)


def test_extract_date_from_filename_no_date():
    assert po.extract_date_from_filename(Path("DSC_0001.jpg")) is None


def test_extract_date_from_filename_rejects_out_of_range_year():
    # 0001-02-03 is outside MIN_VALID_YEAR..MAX_VALID_YEAR, so it should be rejected
    assert po.extract_date_from_filename(Path("00010203_photo.jpg")) is None


# ── is_date_suspicious ──────────────────────────────────────────────────────

def test_is_date_suspicious_none():
    assert po.is_date_suspicious(None) is True


def test_is_date_suspicious_in_range():
    assert po.is_date_suspicious(datetime(2020, 1, 1)) is False


def test_is_date_suspicious_out_of_range():
    assert po.is_date_suspicious(datetime(1970, 1, 1)) is True


# ── find_duplicates ──────────────────────────────────────────────────────────

def make_record(path, file_hash=None, phash=None, date=None, size=100):
    return po.PhotoRecord(path=path, file_hash=file_hash, phash=phash, date=date, size=size)


def test_find_duplicates_exact_hash_keeps_oldest():
    older = make_record("/a/old.jpg", file_hash="abc", date=datetime(2019, 1, 1))
    newer = make_record("/a/new.jpg", file_hash="abc", date=datetime(2020, 1, 1))
    dup_groups = po.find_duplicates([older, newer], hash_threshold=4)
    assert dup_groups == {"/a/old.jpg": ["/a/new.jpg"]}


def test_find_duplicates_no_duplicates():
    a = make_record("/a/1.jpg", file_hash="aaa")
    b = make_record("/a/2.jpg", file_hash="bbb")
    dup_groups = po.find_duplicates([a, b], hash_threshold=4)
    assert dup_groups == {}


def test_find_duplicates_falls_back_to_larger_file_when_no_date():
    small = make_record("/a/small.jpg", file_hash="xyz", date=None, size=10)
    big = make_record("/a/big.jpg", file_hash="xyz", date=None, size=1000)
    dup_groups = po.find_duplicates([small, big], hash_threshold=4)
    # Neither has a date, so the larger file should be kept.
    assert dup_groups == {"/a/big.jpg": ["/a/small.jpg"]}


# ── BKTree ───────────────────────────────────────────────────────────────────

def test_bktree_query_finds_only_entries_within_radius():
    tree = po.BKTree()
    h0 = imagehash.hex_to_hash("0" * 16)
    h1 = imagehash.hex_to_hash("0" * 15 + "1")   # 1 bit set -> distance 1 from h0
    h5 = imagehash.hex_to_hash("0" * 14 + "1f")  # 5 bits set -> distance 5 from h0
    tree.add("a", h0)
    tree.add("b", h1)
    tree.add("c", h5)

    found = {path for path, _dist in tree.query(h0, radius=2)}

    assert found == {"a", "b"}


# ── find_duplicates visual (perceptual-hash) clustering ─────────────────────

def test_find_duplicates_visual_match_via_perceptual_hash():
    hash_a = str(imagehash.hex_to_hash("0" * 16))
    hash_b = str(imagehash.hex_to_hash("0" * 15 + "3"))  # distance 2 from a

    a = make_record("/x/a.jpg", file_hash="ha", phash=hash_a, date=datetime(2020, 1, 1))
    b = make_record("/x/b.jpg", file_hash="hb", phash=hash_b, date=datetime(2020, 1, 5))

    dup_groups = po.find_duplicates([a, b], hash_threshold=4)

    assert dup_groups == {"/x/a.jpg": ["/x/b.jpg"]}


def test_find_duplicates_visual_clustering_is_greedy_not_transitive():
    # a<->b are within threshold, b<->c are within threshold, but a<->c are not.
    # The original O(n^2) implementation claims matches for the first unclaimed
    # "leader" it sees and never re-merges across leaders, so c ends up alone
    # rather than transitively joining a's group. The BK-tree-backed rewrite
    # must reproduce this exact behavior, not full transitive-closure clustering.
    hash_a = str(imagehash.hex_to_hash("0" * 16))
    hash_b = str(imagehash.hex_to_hash("0" * 15 + "7"))    # distance 3 from a
    hash_c = str(imagehash.hex_to_hash("0" * 14 + "3f"))   # distance 3 from b, 6 from a

    a = make_record("/x/a.jpg", file_hash="ha", phash=hash_a, date=datetime(2020, 1, 1))
    b = make_record("/x/b.jpg", file_hash="hb", phash=hash_b, date=datetime(2020, 1, 2))
    c = make_record("/x/c.jpg", file_hash="hc", phash=hash_c, date=datetime(2020, 1, 3))

    dup_groups = po.find_duplicates([a, b, c], hash_threshold=4)

    assert dup_groups == {"/x/a.jpg": ["/x/b.jpg"]}


# ── safe_copy ────────────────────────────────────────────────────────────────

def test_safe_copy_basic(tmp_path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"hello world")
    dest = tmp_path / "out" / "src.jpg"

    result = po.safe_copy(src, dest)

    assert result == dest
    assert dest.read_bytes() == b"hello world"


def test_safe_copy_collision_appends_suffix(tmp_path):
    src1 = tmp_path / "src1.jpg"
    src1.write_bytes(b"first")
    src2 = tmp_path / "src2.jpg"
    src2.write_bytes(b"second")

    dest = tmp_path / "out" / "photo.jpg"
    result1 = po.safe_copy(src1, dest)
    result2 = po.safe_copy(src2, dest)

    assert result1 == dest
    assert result2 == tmp_path / "out" / "photo_1.jpg"
    assert result1.read_bytes() == b"first"
    assert result2.read_bytes() == b"second"


def test_safe_copy_never_overwrites_existing_file(tmp_path):
    dest = tmp_path / "out" / "photo.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"do not touch")

    src = tmp_path / "new.jpg"
    src.write_bytes(b"new content")

    result = po.safe_copy(src, dest)

    assert result == tmp_path / "out" / "photo_1.jpg"
    assert dest.read_bytes() == b"do not touch"


# ── scan_photos symlink handling ────────────────────────────────────────────

def test_scan_photos_skips_symlinked_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    real_target_dir = tmp_path / "outside"
    real_target_dir.mkdir()
    outside_file = real_target_dir / "secret.jpg"
    outside_file.write_bytes(b"not a real photo")

    inside_file = source / "real.jpg"
    inside_file.write_bytes(b"real photo")

    symlink_file = source / "link.jpg"
    symlink_file.symlink_to(outside_file)

    found = po.scan_photos(source)

    assert inside_file in found
    assert symlink_file not in found


def test_scan_photos_skips_symlinked_directories(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.jpg").write_bytes(b"secret")

    (source / "linked_dir").symlink_to(outside_dir, target_is_directory=True)

    found = po.scan_photos(source)

    assert found == []


# ── validate_paths ───────────────────────────────────────────────────────────

def test_validate_paths_rejects_output_inside_source(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    output = source / "output"

    try:
        po.validate_paths(source, output)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1


def test_validate_paths_accepts_sibling_dirs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"

    resolved_source, resolved_output = po.validate_paths(source, output)

    assert resolved_source == source.resolve()
    assert resolved_output == output.resolve()


# ── resolve_log_path ─────────────────────────────────────────────────────────

def test_resolve_log_path_default(tmp_path):
    output = tmp_path / "output"
    assert po.resolve_log_path(None, output) == output / "photo_organizer.log"


def test_resolve_log_path_explicit(tmp_path):
    output = tmp_path / "output"
    custom = tmp_path / "custom.log"
    assert po.resolve_log_path(str(custom), output) == custom


# ── setup_logging ────────────────────────────────────────────────────────────

def test_setup_logging_writes_file_and_captures_warnings(tmp_path):
    log_path = tmp_path / "run.log"
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        po.setup_logging(log_path)
        warnings.warn("test warning message", UserWarning)
        for h in root.handlers:
            h.flush()
        content = log_path.read_text(encoding="utf-8")
        assert "test warning message" in content
    finally:
        for h in list(root.handlers):
            if h not in original_handlers:
                root.removeHandler(h)
                h.close()
        root.setLevel(original_level)
        logging.captureWarnings(False)


# ── compute_perceptual_hash ──────────────────────────────────────────────────

def test_compute_perceptual_hash_no_warning_on_palette_transparency(tmp_path):
    img = Image.new("P", (8, 8))
    img.info["transparency"] = 0
    path = tmp_path / "palette.png"
    img.save(path, format="PNG")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = po.compute_perceptual_hash(path)

    assert result is not None
    assert not any("Palette images with Transparency" in str(w.message) for w in caught)


# ── build_report_html error overflow ────────────────────────────────────────

def test_build_report_html_shows_error_overflow_count():
    errors = [{"path": f"/a/{i}.jpg", "error": "boom"} for i in range(po.REPORT_MAX_ERROR_ROWS + 5)]
    results = po.OrganizeResults(total=205, organized=0, duplicates=0, errors=errors)
    html = po.build_report_html(results, Path("/out"), dry_run=False, elapsed=1.0)
    assert "...and 5 more" in html


# ── organize_photos errors/ folder ──────────────────────────────────────────

def test_organize_photos_copies_errored_photo_to_errors_folder(tmp_path):
    src = tmp_path / "bad.jpg"
    src.write_bytes(b"not really a photo")
    output = tmp_path / "out"

    record = make_record(str(src))
    record.error = "could not read EXIF"

    results = po.organize_photos([record], {}, output, dry_run=False)

    assert (output / "errors" / "bad.jpg").read_bytes() == b"not really a photo"
    assert results.errors_copied == 1


def test_organize_photos_dry_run_does_not_copy_errored_photo(tmp_path):
    src = tmp_path / "bad.jpg"
    src.write_bytes(b"not really a photo")
    output = tmp_path / "out"

    record = make_record(str(src))
    record.error = "could not read EXIF"

    results = po.organize_photos([record], {}, output, dry_run=True)

    assert not (output / "errors").exists()
    assert results.errors_copied == 0
