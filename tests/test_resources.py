from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from paddleocr_vl_rocm.resources import (
    Resource,
    download_resource,
    load_runtime_manifest,
    verify_resource,
)


class FakeResponse:
    def __init__(self, status_code: int, content: bytes, headers=None) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        yield from (
            self.content[index : index + chunk_size]
            for index in range(0, len(self.content), chunk_size)
        )

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, *, headers, stream, timeout):
        self.calls.append({"url": url, "headers": headers, "stream": stream, "timeout": timeout})
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def resource_for(content: bytes, destination: str = "models/asset.bin") -> Resource:
    return Resource(
        name="asset",
        url="https://example.test/asset.bin",
        destination=destination,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_verify_resource_accepts_exact_bytes(tmp_path):
    content = b"verified asset"
    path = tmp_path / "asset.bin"
    path.write_bytes(content)

    verify_resource(path, resource_for(content))


@pytest.mark.parametrize("failure", ["size", "sha256"])
def test_verify_resource_rejects_mismatch(tmp_path, failure):
    expected = b"expected"
    path = tmp_path / "asset.bin"
    path.write_bytes(b"wrong" if failure == "size" else b"expacted")

    with pytest.raises(RuntimeError, match="Size mismatch|SHA-256 mismatch"):
        verify_resource(path, resource_for(expected))


def test_download_resource_resumes_partial_file(tmp_path):
    content = b"abcdefgh"
    resource = resource_for(content)
    part = tmp_path / f"{resource.destination}.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(content[:3])
    response = FakeResponse(206, content[3:], {"Content-Range": "bytes 3-7/8"})
    session = FakeSession(response)

    destination = download_resource(resource, tmp_path, session=session)

    assert destination.read_bytes() == content
    assert session.calls[0]["headers"] == {"Range": "bytes=3-"}
    assert response.closed is True
    assert not part.exists()


def test_download_resource_restarts_when_server_ignores_range(tmp_path):
    content = b"abcdefgh"
    resource = resource_for(content)
    part = tmp_path / f"{resource.destination}.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"abc")
    session = FakeSession(FakeResponse(200, content))

    destination = download_resource(resource, tmp_path, session=session)

    assert destination.read_bytes() == content


def test_download_resource_restarts_on_mismatched_content_range(tmp_path):
    content = b"abcdefgh"
    resource = resource_for(content)
    part = tmp_path / f"{resource.destination}.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(content[:3])
    bad_range = FakeResponse(206, b"wrong", {"Content-Range": "bytes 4-7/8"})
    full = FakeResponse(200, content)
    session = FakeSession(bad_range, full)

    destination = download_resource(resource, tmp_path, session=session)

    assert destination.read_bytes() == content
    assert session.calls[1]["headers"] == {}
    assert bad_range.closed is True
    assert full.closed is True


def test_download_resource_recovers_from_range_not_satisfiable(tmp_path):
    content = b"abcdefgh"
    resource = resource_for(content)
    part = tmp_path / f"{resource.destination}.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"oversized partial")
    rejected = FakeResponse(416, b"")
    full = FakeResponse(200, content)
    session = FakeSession(rejected, full)

    destination = download_resource(resource, tmp_path, session=session)

    assert destination.read_bytes() == content
    assert session.calls[1]["headers"] == {}
    assert rejected.closed is True
    assert full.closed is True


def test_download_resource_reports_progress(tmp_path):
    content = b"abcdefgh"
    resource = resource_for(content)
    progress = []

    download_resource(
        resource,
        tmp_path,
        session=FakeSession(FakeResponse(200, content)),
        progress=lambda downloaded, total: progress.append((downloaded, total)),
    )

    assert progress[-1] == (len(content), len(content))


def test_download_resource_rejects_parent_destination(tmp_path):
    resource = resource_for(b"content", destination="../outside.bin")

    with pytest.raises(ValueError, match="Unsafe destination"):
        download_resource(resource, tmp_path, session=FakeSession(FakeResponse(200, b"content")))


def test_checksum_failure_removes_part_and_preserves_existing_destination(tmp_path):
    content = b"expected"
    resource = resource_for(content)
    destination = tmp_path / resource.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"previous verified asset")
    session = FakeSession(FakeResponse(200, b"corrupt!"))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_resource(resource, tmp_path, session=session)

    assert destination.read_bytes() == b"previous verified asset"
    assert not Path(f"{destination}.part").exists()


def test_packaged_runtime_manifest_loads_five_resources():
    manifest = load_runtime_manifest()

    assert manifest["schema"] == 1
    assert len(manifest["resources"]) == 5


@pytest.mark.parametrize(
    "destination",
    ["", ".", r"..\outside.bin", r"C:\outside.bin", r"\\server\share\outside.bin"],
)
def test_download_resource_rejects_windows_escape_paths(tmp_path, destination):
    resource = resource_for(b"content", destination=destination)

    with pytest.raises(ValueError, match="Unsafe destination"):
        download_resource(resource, tmp_path, session=FakeSession(FakeResponse(200, b"content")))


def test_download_resource_closes_owned_session(tmp_path, monkeypatch):
    content = b"content"
    session = FakeSession(FakeResponse(200, content))
    monkeypatch.setattr("paddleocr_vl_rocm.resources.requests.Session", lambda: session)

    download_resource(resource_for(content), tmp_path)

    assert session.closed is True
