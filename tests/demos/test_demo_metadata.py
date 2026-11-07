import json
import os

import pytest

from tools.demos.metadata import write_metadata


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_identical_metadata_preserves_bytes_and_mtime(tmp_path, newline):
    path = tmp_path / "meta.json"
    content = (json.dumps({"count": 12}, indent=2) + "\n").replace("\n", newline)
    path.write_bytes(content.encode())
    os.utime(path, ns=(1_600_000_000_000_000_000,) * 2)
    before = path.stat().st_mtime_ns
    write_metadata(path, {"count": 12})
    assert path.read_bytes() == content.encode()
    assert path.stat().st_mtime_ns == before


def test_new_and_changed_metadata_are_written(tmp_path):
    path = tmp_path / "meta.json"
    write_metadata(path, {"count": 12})
    assert json.loads(path.read_text()) == {"count": 12}
    write_metadata(path, {"count": 13})
    assert json.loads(path.read_text()) == {"count": 13}
    assert b"\r" not in path.read_bytes()
