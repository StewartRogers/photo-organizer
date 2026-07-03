import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import photo_organizer as po


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
