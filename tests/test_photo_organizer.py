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

    result, error = po.safe_copy(src, dest)

    assert result == dest
    assert error is None
    assert dest.read_bytes() == b"hello world"


def test_safe_copy_collision_appends_suffix(tmp_path):
    src1 = tmp_path / "src1.jpg"
    src1.write_bytes(b"first")
    src2 = tmp_path / "src2.jpg"
    src2.write_bytes(b"second")

    dest = tmp_path / "out" / "photo.jpg"
    result1, error1 = po.safe_copy(src1, dest)
    result2, error2 = po.safe_copy(src2, dest)

    assert result1 == dest
    assert result2 == tmp_path / "out" / "photo_1.jpg"
    assert error1 is None
    assert error2 is None
    assert result1.read_bytes() == b"first"
    assert result2.read_bytes() == b"second"


def test_safe_copy_never_overwrites_existing_file(tmp_path):
    dest = tmp_path / "out" / "photo.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"do not touch")

    src = tmp_path / "new.jpg"
    src.write_bytes(b"new content")

    result, error = po.safe_copy(src, dest)

    assert result == tmp_path / "out" / "photo_1.jpg"
    assert error is None
    assert dest.read_bytes() == b"do not touch"


def test_safe_copy_returns_reason_when_dest_dir_is_a_file(tmp_path):
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_bytes(b"x")
    src = tmp_path / "src.jpg"
    src.write_bytes(b"hello")
    dest = blocking_file / "src.jpg"

    result, error = po.safe_copy(src, dest)

    assert result is None
    assert error is not None


# ── scan_photos symlink handling ────────────────────────────────────────────

def test_scan_photos_skips_symlinked_files(tmp_path, monkeypatch):
    # Creating a real symlink requires Developer Mode / admin elevation on
    # Windows (SeCreateSymbolicLinkPrivilege) — not available in every dev/CI
    # environment. scan_photos()'s entire symlink guard is a Path.is_symlink()
    # check, so simulating that directly tests the real protective logic
    # without depending on OS-level symlink-creation privileges anywhere.
    source = tmp_path / "source"
    source.mkdir()

    inside_file = source / "real.jpg"
    inside_file.write_bytes(b"real photo")

    symlink_file = source / "link.jpg"
    symlink_file.write_bytes(b"stand-in for a symlink pointing outside source")

    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink",
                         lambda self: self == symlink_file or real_is_symlink(self))

    found = po.scan_photos(source)

    assert inside_file in found
    assert symlink_file not in found


def test_scan_photos_skips_symlinked_directories(tmp_path, monkeypatch):
    # See test_scan_photos_skips_symlinked_files: simulates a symlinked
    # directory via Path.is_symlink() rather than requiring real
    # symlink-creation privileges.
    source = tmp_path / "source"
    source.mkdir()
    linked_dir = source / "linked_dir"
    linked_dir.mkdir()
    (linked_dir / "secret.jpg").write_bytes(b"secret")

    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink",
                         lambda self: self == linked_dir or real_is_symlink(self))

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


def test_organize_photos_records_real_copy_failure_reason(tmp_path, monkeypatch):
    record = make_record("/a/photo.jpg", file_hash="abc", date=datetime(2020, 1, 1))
    monkeypatch.setattr(po, "safe_copy", lambda src, dest: (None, "disk full"))

    results = po.organize_photos([record], {}, tmp_path / "out", dry_run=False)

    assert len(results.errors) == 1
    assert "disk full" in results.errors[0]["error"]


# ── is_cloud_only_placeholder ────────────────────────────────────────────────

class _FakeStat:
    def __init__(self, attrs):
        self.st_file_attributes = attrs


def test_is_cloud_only_placeholder_detects_recall_bit():
    assert po.is_cloud_only_placeholder(_FakeStat(po.FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)) is True


def test_is_cloud_only_placeholder_false_for_normal_file():
    assert po.is_cloud_only_placeholder(_FakeStat(0)) is False


def test_is_cloud_only_placeholder_false_when_attribute_missing():
    class NoAttrs:
        pass
    assert po.is_cloud_only_placeholder(NoAttrs()) is False


# ── process_photo cloud-only short-circuit ──────────────────────────────────

def test_process_photo_marks_cloud_only_without_hashing(tmp_path, monkeypatch):
    photo = tmp_path / "cloud.jpg"
    photo.write_bytes(b"fake image data")

    class FakeCloudStat:
        st_size = 123
        st_file_attributes = po.FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS

    monkeypatch.setattr(po.Path, "stat", lambda self: FakeCloudStat())

    record = po.process_photo(photo)

    assert record.cloud_only is True
    assert record.error is None
    assert record.file_hash is None


def test_hydrate_cloud_only_file_reads_successfully(tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"some content")

    assert po.hydrate_cloud_only_file(photo) is True


def test_hydrate_cloud_only_file_returns_false_on_read_failure(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    assert po.hydrate_cloud_only_file(missing) is False


def test_process_photo_hydrates_and_processes_cloud_only_file_when_requested(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_20190704_120000.jpg"
    photo.write_bytes(b"fake image data")

    class FakeCloudStat:
        st_size = 123
        st_file_attributes = po.FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS

    monkeypatch.setattr(po.Path, "stat", lambda self: FakeCloudStat())
    monkeypatch.setattr(po, "hydrate_cloud_only_file", lambda path: True)

    record = po.process_photo(photo, hydrate=True)

    assert record.cloud_only is False
    assert record.error is None
    assert record.date == datetime(2019, 7, 4, 12, 0, 0)  # normal processing ran


def test_process_photo_falls_back_to_cloud_only_when_hydration_fails(tmp_path, monkeypatch):
    photo = tmp_path / "cloud.jpg"
    photo.write_bytes(b"fake image data")

    class FakeCloudStat:
        st_size = 123
        st_file_attributes = po.FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS

    monkeypatch.setattr(po.Path, "stat", lambda self: FakeCloudStat())
    monkeypatch.setattr(po, "hydrate_cloud_only_file", lambda path: False)

    record = po.process_photo(photo, hydrate=True)

    assert record.cloud_only is True
    assert record.file_hash is None


# ── organize_photos cloud-only handling ─────────────────────────────────────

def test_organize_photos_skips_cloud_only_without_copying(tmp_path):
    record = make_record("/a/cloud.jpg")
    record.cloud_only = True
    output = tmp_path / "out"

    results = po.organize_photos([record], {}, output, dry_run=False)

    assert results.cloud_only == ["/a/cloud.jpg"]
    assert results.errors == []
    assert not (output / "errors").exists()
    assert not (output / "organized").exists()


def test_organize_photos_excludes_cloud_only_from_suspicious_dates(tmp_path):
    # Cloud-only photos never reach get_photo_date(), so date stays None —
    # they must not also show up in the suspicious-dates bucket.
    record = make_record("/a/cloud.jpg")
    record.cloud_only = True
    assert record.date is None

    results = po.organize_photos([record], {}, tmp_path / "out", dry_run=False)

    assert results.suspicious_dates == []
    assert results.cloud_only == ["/a/cloud.jpg"]


# ── retry file round-trip ────────────────────────────────────────────────────

def test_write_retry_file_and_load_retry_paths_roundtrip(tmp_path):
    results = po.OrganizeResults(
        total=2, organized=0, duplicates=0,
        errors=[{"path": "/a/broken.jpg", "error": "boom"}],
        cloud_only=["/a/cloud.jpg"],
    )

    retry_path = po.write_retry_file(results, tmp_path, run_id="20260704_101423")

    assert retry_path == tmp_path / "retry_photos_20260704_101423.txt"
    assert po.load_retry_paths(retry_path) == [Path("/a/broken.jpg"), Path("/a/cloud.jpg")]


def test_write_retry_file_returns_none_when_nothing_to_retry(tmp_path):
    results = po.OrganizeResults(total=1, organized=1, duplicates=0)
    assert po.write_retry_file(results, tmp_path, run_id="20260704_101423") is None


def test_write_retry_file_uses_distinct_names_per_run(tmp_path):
    results = po.OrganizeResults(
        total=1, organized=0, duplicates=0,
        errors=[{"path": "/a/broken.jpg", "error": "boom"}],
    )

    first = po.write_retry_file(results, tmp_path, run_id="20260704_101423")
    second = po.write_retry_file(results, tmp_path, run_id="20260704_111500")

    assert first != second
    assert first.exists() and second.exists()


# ── estimate_required_bytes / check_disk_space ──────────────────────────────

def test_estimate_required_bytes_sums_regular_files(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"x" * 100)
    b = tmp_path / "b.jpg"
    b.write_bytes(b"y" * 250)

    needed, cloud_only = po.estimate_required_bytes([a, b])

    assert needed == 350
    assert cloud_only == 0


def test_estimate_required_bytes_separates_cloud_only(tmp_path, monkeypatch):
    normal = tmp_path / "normal.jpg"
    normal.write_bytes(b"x" * 100)
    cloud = tmp_path / "cloud.jpg"
    cloud.write_bytes(b"y" * 50)

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        st = real_stat(self, *args, **kwargs)
        if self.name == "cloud.jpg":
            class FakeStat:
                st_size = st.st_size
                st_file_attributes = po.FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
            return FakeStat()
        return st

    monkeypatch.setattr(po.Path, "stat", fake_stat)

    needed, cloud_only = po.estimate_required_bytes([normal, cloud])

    assert needed == 100
    assert cloud_only == 50


def test_estimate_required_bytes_skips_missing_files(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    needed, cloud_only = po.estimate_required_bytes([missing])
    assert needed == 0
    assert cloud_only == 0


def test_check_disk_space_passes_when_enough_free(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    class FakeUsage:
        free = 10 * 1024 ** 3

    monkeypatch.setattr(po.shutil, "disk_usage", lambda path: FakeUsage())

    po.check_disk_space([photo], tmp_path / "out", dry_run=False)  # should not raise


def test_check_disk_space_exits_when_insufficient_and_not_dry_run(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (5 * 1024 ** 3, 0))

    class FakeUsage:
        free = 0

    monkeypatch.setattr(po.shutil, "disk_usage", lambda path: FakeUsage())

    try:
        po.check_disk_space([photo], tmp_path / "out", dry_run=False)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1


def test_check_disk_space_warns_but_does_not_exit_during_dry_run(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (5 * 1024 ** 3, 0))

    class FakeUsage:
        free = 0

    monkeypatch.setattr(po.shutil, "disk_usage", lambda path: FakeUsage())

    po.check_disk_space([photo], tmp_path / "out", dry_run=True)  # should not raise


# ── check_cloud_only_backlog ─────────────────────────────────────────────────

def test_check_cloud_only_backlog_passes_when_under_threshold(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (100, 0))

    assert po.check_cloud_only_backlog([photo], max_cloud_only_gb=1.0, dry_run=False) is False


def test_check_cloud_only_backlog_returns_true_to_hydrate_small_backlog(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (0, 1024))  # small, but > 0

    assert po.check_cloud_only_backlog([photo], max_cloud_only_gb=1.0, dry_run=False) is True


def test_check_cloud_only_backlog_never_hydrates_during_dry_run(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (0, 1024))  # small, but > 0

    assert po.check_cloud_only_backlog([photo], max_cloud_only_gb=1.0, dry_run=True) is False


def test_check_cloud_only_backlog_exits_when_over_threshold_and_not_dry_run(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (0, 5 * 1024 ** 3))

    try:
        po.check_cloud_only_backlog([photo], max_cloud_only_gb=1.0, dry_run=False)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1


def test_check_cloud_only_backlog_warns_but_does_not_exit_during_dry_run(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x" * 100)

    monkeypatch.setattr(po, "estimate_required_bytes", lambda paths: (0, 5 * 1024 ** 3))

    assert po.check_cloud_only_backlog([photo], max_cloud_only_gb=1.0, dry_run=True) is False


# ── video support ────────────────────────────────────────────────────────────

def test_is_video_detects_supported_extensions():
    assert po.is_video(Path("clip.mp4")) is True
    assert po.is_video(Path("clip.MOV")) is True
    assert po.is_video(Path("photo.jpg")) is False


def test_scan_photos_finds_video_files(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"fake video data")
    (tmp_path / "photo.jpg").write_bytes(b"fake image data")
    (tmp_path / "notes.txt").write_bytes(b"not media")

    found = po.scan_photos(tmp_path)

    names = {p.name for p in found}
    assert names == {"clip.mp4", "photo.jpg"}


def test_process_photo_skips_hashing_for_video_but_still_gets_a_date(tmp_path):
    video = tmp_path / "VID_20190704_120000.mp4"
    video.write_bytes(b"fake video data")

    record = po.process_photo(video)

    assert record.error is None
    assert record.file_hash is None
    assert record.phash is None
    assert record.date == datetime(2019, 7, 4, 12, 0, 0)
    assert record.date_source == "filename"


# ── get_video_date ───────────────────────────────────────────────────────────

class _FakeVideoStream:
    def close(self):
        pass


class _FakeVideoParser:
    stream = _FakeVideoStream()


def _fake_metadata(creation_date):
    class FakeMetadata:
        def get(self, key, default=None):
            if key == "creation_date":
                return creation_date
            return default
    return FakeMetadata()


def test_get_video_date_prefers_container_metadata(tmp_path, monkeypatch):
    # Filename has no date at all, so metadata is the only possible source.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    monkeypatch.setattr(po, "VIDEO_METADATA_SUPPORTED", True)
    monkeypatch.setattr(po, "createParser", lambda path: _FakeVideoParser())
    monkeypatch.setattr(po, "extractMetadata",
                         lambda parser: _fake_metadata(datetime(2022, 4, 7, 17, 48, 1)))

    dt, source = po.get_video_date(video)

    assert dt == datetime(2022, 4, 7, 17, 48, 1)
    assert source == "video metadata"


def test_get_video_date_falls_back_to_filename_when_metadata_unavailable(tmp_path, monkeypatch):
    video = tmp_path / "VID_20190704_120000.mp4"
    video.write_bytes(b"fake video bytes")

    monkeypatch.setattr(po, "VIDEO_METADATA_SUPPORTED", True)
    monkeypatch.setattr(po, "createParser", lambda path: None)  # unparseable

    dt, source = po.get_video_date(video)

    assert dt == datetime(2019, 7, 4, 12, 0, 0)
    assert source == "filename"


def test_get_video_date_prefers_filesystem_over_suspicious_metadata(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"  # no filename date
    video.write_bytes(b"fake video bytes")

    monkeypatch.setattr(po, "VIDEO_METADATA_SUPPORTED", True)
    monkeypatch.setattr(po, "createParser", lambda path: _FakeVideoParser())
    monkeypatch.setattr(po, "extractMetadata",
                         lambda parser: _fake_metadata(datetime(1970, 1, 1)))  # suspicious

    dt, source = po.get_video_date(video)

    assert dt is not None
    assert "filesystem" in source
    assert "video metadata suspicious" in source


def test_get_video_date_skips_metadata_lookup_when_hachoir_not_installed(tmp_path, monkeypatch):
    video = tmp_path / "VID_20190704_120000.mp4"
    video.write_bytes(b"fake video bytes")

    monkeypatch.setattr(po, "VIDEO_METADATA_SUPPORTED", False)

    dt, source = po.get_video_date(video)

    assert dt == datetime(2019, 7, 4, 12, 0, 0)
    assert source == "filename"


def test_find_duplicates_never_groups_identical_videos():
    # Same content, same size, but videos never get a file_hash/phash, so
    # find_duplicates() must not group them even though the bytes match.
    a = make_record("/a/clip1.mp4", date=datetime(2020, 1, 1))
    b = make_record("/a/clip2.mp4", date=datetime(2020, 1, 1))

    dup_groups = po.find_duplicates([a, b], hash_threshold=4)

    assert dup_groups == {}


def test_organize_photos_organizes_video_like_a_photo(tmp_path):
    video_src = tmp_path / "VID_20190704_120000.mp4"
    video_src.write_bytes(b"fake video data")
    record = make_record(str(video_src), date=datetime(2019, 7, 4))

    output = tmp_path / "out"
    results = po.organize_photos([record], {}, output, dry_run=False)

    assert results.organized == 1
    assert results.duplicates == 0
    assert (output / "organized" / "2019" / "07" / "VID_20190704_120000.mp4").exists()


# ── archive_source / _verify_zip ─────────────────────────────────────────────

def test_verify_zip_returns_none_for_valid_zip(tmp_path):
    import zipfile
    zip_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("hello.txt", b"hello world")

    assert po._verify_zip(zip_path) is None


def test_verify_zip_detects_corruption(tmp_path):
    import zipfile
    zip_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("hello.txt", b"hello world" * 100)

    # Flip some bytes in the middle of the file's raw data to break its CRC
    # (or, depending on offset, the archive structure itself) — either way
    # _verify_zip() must report it, not silently pass.
    data = bytearray(zip_path.read_bytes())
    mid = len(data) // 2
    data[mid] = data[mid] ^ 0xFF
    zip_path.write_bytes(bytes(data))

    assert po._verify_zip(zip_path) is not None


def test_archive_source_creates_verified_zip_with_relative_paths(tmp_path):
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "a.jpg").write_bytes(b"photo a")
    (source / "sub" / "b.jpg").write_bytes(b"photo b")

    zip_paths, problems = po.archive_source(source, chunk_gb=1.0, run_id="20260705_120000")

    assert problems == []
    assert len(zip_paths) == 1
    assert zip_paths[0].parent == source
    assert zip_paths[0].name == "archive_20260705_120000_0001.zip"

    import zipfile
    with zipfile.ZipFile(zip_paths[0]) as zf:
        names = set(zf.namelist())
        assert "a.jpg" in names
        assert str(Path("sub") / "b.jpg") in names or "sub/b.jpg" in names
        assert zf.read("a.jpg") == b"photo a"


def test_archive_source_splits_into_multiple_chunks_by_size(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(5):
        (source / f"file_{i}.jpg").write_bytes(b"x" * 1000)

    # ~1000 bytes/file; a ~2500 byte chunk should force multiple archives
    zip_paths, problems = po.archive_source(source, chunk_gb=2500 / (1024 ** 3), run_id="run1")

    assert problems == []
    assert len(zip_paths) > 1

    import zipfile
    total_members = 0
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            assert zf.testzip() is None
            total_members += len(zf.namelist())
    assert total_members == 5


def test_archive_source_does_not_reinclude_existing_archives(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "photo.jpg").write_bytes(b"photo")
    (source / "archive_oldrun_0001.zip").write_bytes(b"pretend old archive")

    zip_paths, problems = po.archive_source(source, chunk_gb=1.0, run_id="newrun")

    assert problems == []
    import zipfile
    with zipfile.ZipFile(zip_paths[0]) as zf:
        assert zf.namelist() == ["photo.jpg"]


def test_archive_source_reports_verification_failures(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "photo.jpg").write_bytes(b"photo")

    monkeypatch.setattr(po, "_verify_zip", lambda path: "simulated corruption")

    zip_paths, problems = po.archive_source(source, chunk_gb=1.0, run_id="run1")

    assert len(zip_paths) == 1
    assert problems == [(str(zip_paths[0]), "simulated corruption")]
