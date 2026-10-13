"""Public downloads plus explicitly supplied, authorized training-file URLs."""
from __future__ import annotations

import hashlib
import base64
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .io import sha256, write_json


RESERVE_BYTES = 1024 ** 3


def resources_path(name):
    return Path(__file__).parent / "resources" / name


def load_catalog():
    return json.loads(resources_path("data-catalog.json").read_text())


def relative_path(root, name):
    root = Path(root).resolve()
    destination = (root / name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Data/model destinations must stay under the requested root")
    return destination


def merge_manifest(catalog, path):
    if not path:
        return catalog
    overrides = json.loads(Path(path).read_text())
    if isinstance(overrides, dict):
        overrides = overrides["files"]
    known = {entry["id"]: entry for entry in catalog["files"]}
    for item in overrides:
        if item["id"] not in known:
            raise ValueError(f"Unknown catalog id: {item['id']}")
        entry = known[item["id"]]
        if entry["kind"] != "training":
            raise ValueError("Manifest may only provide URLs for catalogued released training files")
        if item.get("authorized") is not True:
            raise ValueError("Training manifest requires authorized:true for each supplied file")
        url = item.get("url")
        local = item.get("local_path")
        if bool(url) == bool(local):
            raise ValueError("Provide exactly one of url/local_path for each training file")
        if url and urlsplit(url).scheme != "https":
            raise ValueError("Authorized remote downloads must use HTTPS")
        entry.update({key: item[key] for key in ("url", "local_path", "sha256", "size_bytes") if key in item})
        entry["authorized"] = True
    return catalog


def validate_download(path, entry):
    size = Path(path).stat().st_size
    if entry.get("size_bytes") is not None and size != int(entry["size_bytes"]):
        raise ValueError(f"Size mismatch: expected {entry['size_bytes']}, got {size}")
    actual = sha256(Path(path))
    if entry.get("sha256") and actual != entry["sha256"]:
        raise ValueError("SHA256 mismatch")
    if str(path).endswith((".h5ad", ".h5ad.partial")):
        with Path(path).open("rb") as stream:
            if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError("Expected HDF5; response may be a login/error page")
    elif entry["path"].endswith(".json"):
        json.loads(Path(path).read_text())
    return {"size_bytes": size, "sha256": actual}


def fetch_file(entry, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and entry.get("kind") != "panel":
        return {"status": "present", **validate_download(destination, entry)}
    partial = destination.with_name(destination.name + ".partial")
    # A complete corrupt previous attempt must not get stuck resuming at EOF.
    if partial.exists() and entry.get("size_bytes") and partial.stat().st_size >= int(entry["size_bytes"]):
        try:
            details = validate_download(partial, entry)
            partial.replace(destination)
            return {"status": "downloaded", **details}
        except ValueError:
            partial.replace(partial.with_name(partial.name + ".corrupt"))
    if entry.get("transport") == "github_blob":
        # Official small samples can also be fetched through GitHub's blob API
        # when raw.githubusercontent.com is unreachable. Hash the decoded bytes.
        if int(entry["size_bytes"]) + RESERVE_BYTES > shutil.disk_usage(destination.parent).free:
            raise ValueError("Insufficient disk space (keeps a 1 GiB reserve)")
        response = requests.get(entry["url"], timeout=(20, 45))
        if response.status_code != 200:
            raise ValueError(f"GitHub blob returned HTTP {response.status_code}")
        blob = response.json()
        if blob.get("encoding") != "base64":
            raise ValueError("Expected base64 GitHub blob")
        partial.write_bytes(base64.b64decode(blob["content"]))
    elif entry.get("local_path"):
        origin = Path(entry["local_path"])
        if origin.stat().st_size + RESERVE_BYTES > shutil.disk_usage(destination.parent).free:
            raise ValueError("Insufficient disk space (keeps a 1 GiB reserve)")
        shutil.copyfile(origin, partial)
    else:
        # Never print signed URLs, cookies or raw requests exception messages.
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("https://", adapter)
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(entry["url"], headers=headers, stream=True, timeout=(20, 45)) as response:
            if response.status_code == 416 and partial.exists():
                total_match = re.fullmatch(r"bytes \*/(\d+)", response.headers.get("Content-Range", ""))
                total = int(total_match[1]) if total_match else entry.get("size_bytes")
                if total is None or partial.stat().st_size != int(total):
                    raise ValueError("Incomplete partial: HTTP416 total does not match local file")
                details = validate_download(partial, entry)
                partial.replace(destination)
                return {"status": "downloaded", **details}
            if response.status_code not in {200, 206}:
                raise ValueError(f"Download returned HTTP {response.status_code}")
            append = response.status_code == 206 and offset > 0
            if response.status_code == 206:
                expected_start = offset if append else 0
                header = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", header)
                if not match or int(match[1]) != expected_start:
                    raise ValueError("Range response does not match requested offset")
                total = None if match[3] == "*" else int(match[3])
                if total is None:
                    total = entry.get("size_bytes")
                    if total is None and not entry.get("sha256"):
                        raise ValueError("Unknown range total requires expected size or SHA256")
                # Some proxies cap a response chunk. Leave partial for the next invocation.
            else:
                total = int(response.headers.get("Content-Length", "0")) or entry.get("size_bytes")
            remaining = max(0, total - (offset if append else 0)) if total else 0
            if remaining + RESERVE_BYTES > shutil.disk_usage(destination.parent).free:
                raise ValueError("Insufficient disk space (keeps a 1 GiB reserve)")
            with partial.open("ab" if append else "wb") as output:
                for block in response.iter_content(1024 * 1024):
                    if not block:
                        continue
                    if shutil.disk_usage(destination.parent).free < len(block) + RESERVE_BYTES:
                        raise ValueError("Disk reserve reached; partial file retained")
                    output.write(block)
            if total and partial.stat().st_size != total:
                raise ValueError("Partial response retained; rerun to resume")
    details = validate_download(partial, entry)
    partial.replace(destination)
    return {"status": "downloaded", **details}


def download_all(*, root=".", manifest=None, task=None, public_only=False):
    catalog = merge_manifest(load_catalog(), manifest)
    rows = []
    for entry in catalog["files"]:
        if task and task not in entry.get("tasks", []):
            continue
        if public_only and entry["kind"] == "training":
            continue
        destination = relative_path(root, entry["path"])
        row = {"id": entry["id"], "kind": entry["kind"], "path": entry["path"]}
        if not entry.get("url") and not entry.get("local_path"):
            row.update(status="blocked", reason="Official registration/download URL required")
        else:
            try:
                row.update(fetch_file(entry, destination))
            except requests.RequestException:
                row.update(status="failed", reason="Network request failed; retry download (URL redacted)")
            except (ValueError, OSError) as exc:
                row.update(status="failed", reason=str(exc))
        rows.append(row)
        print(f"{row['status']:12} {row['id']}", flush=True)
    report = {
        "catalog_version": catalog["version"], "files": rows,
        "public_complete": all(row["status"] in {"downloaded", "present"} for row in rows if row["kind"] != "training"),
        "selected_resources_complete": all(row["status"] in {"downloaded", "present"} for row in rows),
        "note": "Public demo samples are not the complete training data; hidden ground truth is never downloaded.",
    }
    write_json(relative_path(root, "artifacts/download-report.json"), report)
    return report


def model_registry():
    return json.loads(resources_path("model-registry.json").read_text())


def source_hashes(directory):
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != ".source-lock.json"
        and ".git" not in path.relative_to(directory).parts
        and "__pycache__" not in path.relative_to(directory).parts
    }


def install_model(name, root="."):
    registry = model_registry()["external"]
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown external model {name}; see models list")
    entry = registry[key]
    destination = relative_path(root, "third_party/" + key)
    lock = destination / ".source-lock.json"
    if lock.exists():
        record = json.loads(lock.read_text())
        if record["commit"] != entry["commit"]:
            raise ValueError("Existing author source has another lock; use a separate root")
        if "file_sha256" in record and source_hashes(destination) != record["file_sha256"]:
            raise ValueError("Author source changed since acquisition; refusing to report original lock")
        return record
    if destination.exists() and any(path.name != ".git" for path in destination.iterdir()):
        raise ValueError("Existing source directory has files; refusing to overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository = entry["url"].removeprefix("https://github.com/").removesuffix(".git")
    endpoint = f"repos/{repository}/tarball/{entry['commit']}"
    with tempfile.TemporaryDirectory(prefix="embryo-source-", dir=destination.parent) as working:
        archive = Path(working) / "source.tar.gz"
        if shutil.which("gh"):
            with archive.open("wb") as stream:
                process = subprocess.Popen(["gh", "api", endpoint], stdout=subprocess.PIPE)
                try:
                    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
                        if shutil.disk_usage(destination.parent).free < RESERVE_BYTES + len(block):
                            raise ValueError("Disk reserve reached while acquiring author source")
                        stream.write(block)
                    if process.wait(timeout=120):
                        raise ValueError("GitHub source archive request failed")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                    process.stdout.close()
        else:
            with requests.get("https://api.github.com/" + endpoint, stream=True, timeout=(20,45)) as response:
                response.raise_for_status()
                with archive.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if shutil.disk_usage(destination.parent).free < RESERVE_BYTES + len(block):
                            raise ValueError("Disk reserve reached while acquiring author source")
                        stream.write(block)
        expanded = Path(working) / "expanded"
        expanded.mkdir()
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            total = sum(member.size for member in members)
            if total + RESERVE_BYTES > shutil.disk_usage(destination.parent).free:
                raise ValueError("Insufficient disk to extract author source")
            for member in members:
                target = (expanded / member.name).resolve()
                if not target.is_relative_to(expanded) or member.issym() or member.islnk() or member.isdev():
                    raise ValueError("Unsafe path/link in author archive")
            bundle.extractall(expanded)
        folders = list(expanded.iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise ValueError("Unexpected author archive layout")
        destination.mkdir(exist_ok=True)
        for child in folders[0].iterdir():
            shutil.move(str(child), destination / child.name)
        record = {"name": key, "path": str(destination), "commit": entry["commit"],
                  "archive_sha256": sha256(archive), "file_sha256": source_hashes(destination),
                  "status": "source_acquired_not_trained"}
        write_json(lock, record)
    return record
