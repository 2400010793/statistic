"""Download integrity and authorization checks against a local HTTP fixture."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
import requests

from virtual_embryo_benchmark import acquisition


HDF_BYTES = b"\x89HDF\r\n\x1a\n" + bytes(range(32))


@contextmanager
def local_http(*, mode="standard", payload=HDF_BYTES):
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            requests_seen.append(requested)
            start = int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
            if mode == "reject-range" or start >= len(payload):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if mode in {"capped", "unknown-total"}:
                end = min(start + 15, len(payload) - 1)
                status = 206
                total = "*" if mode == "unknown-total" else str(len(payload))
            else:
                end = len(payload) - 1
                status = 206 if requested else 200
                total = str(len(payload))
            body = payload[start:end + 1]
            self.send_response(status)
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/data", requests_seen
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def entry(url, **overrides):
    return {"id": "test", "kind": "training", "path": "data/test.h5ad", "url": url, **overrides}


def test_range_resume_preserves_prefix_and_validates_whole_file(tmp_path):
    destination = tmp_path / "test.h5ad"
    partial = destination.with_suffix(".h5ad.partial")
    partial.write_bytes(HDF_BYTES[:16])
    with local_http() as (url, seen):
        result = acquisition.fetch_file(
            entry(url, size_bytes=len(HDF_BYTES), sha256=hashlib.sha256(HDF_BYTES).hexdigest()), destination,
        )
    assert seen == ["bytes=16-"]
    assert destination.read_bytes() == HDF_BYTES
    assert not partial.exists()
    assert result["status"] == "downloaded"


def test_capped_range_response_remains_partial_then_can_resume(tmp_path):
    destination = tmp_path / "test.h5ad"
    with local_http(mode="capped") as (url, seen):
        download = entry(url, size_bytes=len(HDF_BYTES))
        for _ in range(2):
            with pytest.raises(ValueError, match="Partial response"):
                acquisition.fetch_file(download, destination)
        result = acquisition.fetch_file(download, destination)
    assert seen == [None, "bytes=16-", "bytes=32-"]
    assert result["status"] == "downloaded"
    assert destination.read_bytes() == HDF_BYTES


def test_unknown_total_range_cannot_promote_unverified_fragment(tmp_path):
    destination = tmp_path / "test.h5ad"
    with local_http(mode="unknown-total") as (url, _):
        with pytest.raises(ValueError):
            acquisition.fetch_file(entry(url), destination)
    assert not destination.exists()


def test_416_cannot_promote_partial_smaller_than_server_total(tmp_path):
    destination = tmp_path / "test.h5ad"
    destination.with_suffix(".h5ad.partial").write_bytes(HDF_BYTES[:16])
    with local_http(mode="reject-range") as (url, _):
        with pytest.raises(ValueError):
            acquisition.fetch_file(entry(url), destination)
    assert not destination.exists()


@pytest.mark.parametrize("verification", ["size", "hash"])
def test_size_and_hash_mismatches_never_promote_download(tmp_path, verification):
    destination = tmp_path / "test.h5ad"
    expected = {"size_bytes": len(HDF_BYTES) + 1} if verification == "size" else {"sha256": "0" * 64}
    with local_http() as (url, _):
        with pytest.raises(ValueError, match="mismatch"):
            acquisition.fetch_file(entry(url, **expected), destination)
    assert not destination.exists()


def test_disk_reserve_rejects_transfer_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda path: SimpleNamespace(free=acquisition.RESERVE_BYTES + len(HDF_BYTES) - 1))
    destination = tmp_path / "test.h5ad"
    with local_http() as (url, _):
        with pytest.raises(ValueError, match="Insufficient disk"):
            acquisition.fetch_file(entry(url), destination)
    assert not destination.exists()
    assert not destination.with_suffix(".h5ad.partial").exists()


@pytest.mark.parametrize("name", ["../escape.h5ad", "/tmp/outside-the-data-root.h5ad"])
def test_destinations_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError, match="stay under"):
        acquisition.relative_path(tmp_path, name)


def test_symlink_destination_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="stay under"):
        acquisition.relative_path(root, "data/file.h5ad")


@pytest.mark.parametrize("override", [
    {"id": "held-out", "url": "https://example.org/test.h5ad", "authorized": True},
    {"id": "sample", "url": "https://example.org/sample.h5ad", "authorized": True},
    {"id": "train", "url": "https://example.org/train.h5ad"},
    {"id": "train", "url": "http://example.org/train.h5ad", "authorized": True},
    {"id": "train", "url": "https://example.org/train.h5ad", "local_path": "/tmp/train.h5ad", "authorized": True},
])
def test_manifest_rejects_unknown_nontraining_or_unauthorized_inputs(tmp_path, override):
    catalog = {"files": [{"id": "train", "kind": "training"}, {"id": "sample", "kind": "sample"}]}
    manifest = tmp_path / "private.json"
    manifest.write_text(json.dumps({"files": [override]}))
    with pytest.raises(ValueError):
        acquisition.merge_manifest(catalog, manifest)


def test_network_failure_redacts_signed_url_from_report_and_console(tmp_path, monkeypatch, capsys):
    secret_url = "https://example.org/file?token=private-secret-value"
    catalog = {"version": "test", "files": [{**entry(secret_url), "kind": "sample"}]}
    monkeypatch.setattr(acquisition, "load_catalog", lambda: catalog)

    def failed_request(*args, **kwargs):
        raise requests.ConnectionError("Could not retrieve " + secret_url)

    monkeypatch.setattr(acquisition.requests.Session, "get", failed_request)
    report = acquisition.download_all(root=tmp_path)
    assert report["files"][0]["status"] == "failed"
    assert "private-secret-value" not in json.dumps(report)
    assert "private-secret-value" not in capsys.readouterr().out
