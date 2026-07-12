from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

import requests


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    destination: str
    size: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Resource:
        return cls(
            name=str(value["name"]),
            url=str(value["url"]),
            destination=str(value["destination"]),
            size=int(value["size"]),
            sha256=str(value["sha256"]),
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_resource(path: Path, resource: Resource) -> None:
    if path.stat().st_size != resource.size:
        raise RuntimeError(f"Size mismatch for {resource.name}: {path}")
    if sha256_file(path).lower() != resource.sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {resource.name}: {path}")


def load_runtime_manifest() -> dict[str, Any]:
    manifest = files("paddleocr_vl_rocm").joinpath("assets/runtime-manifest.json")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _destination_path(root: Path, resource: Resource) -> Path:
    if "\\" in resource.destination:
        raise ValueError(f"Unsafe destination for {resource.name}: {resource.destination}")
    relative = PurePosixPath(resource.destination)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe destination for {resource.name}: {resource.destination}")
    resolved_root = root.resolve()
    destination = root.joinpath(*relative.parts)
    resolved_destination = destination.resolve()
    if resolved_destination == resolved_root or not resolved_destination.is_relative_to(
        resolved_root
    ):
        raise ValueError(f"Unsafe destination for {resource.name}: {resource.destination}")
    return destination


def download_resource(
    resource: Resource,
    root: Path,
    session: requests.Session | None = None,
    progress: Callable[[int, int], None] | None = None,
    timeout: float = 300.0,
) -> Path:
    destination = _destination_path(Path(root), resource)
    if destination.is_file():
        try:
            verify_resource(destination, resource)
        except RuntimeError:
            pass
        else:
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")
    if not part.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Unsafe destination for {resource.name}: {resource.destination}")
    if part.is_file():
        try:
            verify_resource(part, resource)
        except RuntimeError:
            pass
        else:
            part.replace(destination)
            return destination
    offset = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    owns_session = session is None
    client = session or requests.Session()
    try:
        response = client.get(resource.url, headers=headers, stream=True, timeout=timeout)
        if offset and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-\d+/(?:\d+|\*)", content_range)
            if match is None or int(match.group(1)) != offset:
                response.close()
                response = client.get(resource.url, headers={}, stream=True, timeout=timeout)
                offset = 0
        elif offset and response.status_code == 416:
            response.close()
            response = client.get(resource.url, headers={}, stream=True, timeout=timeout)
            offset = 0

        try:
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            downloaded = offset if append else 0
            mode = "ab" if append else "wb"
            with part.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, resource.size)
        finally:
            response.close()
    finally:
        if owns_session:
            client.close()

    try:
        verify_resource(part, resource)
    except RuntimeError:
        part.unlink(missing_ok=True)
        raise
    part.replace(destination)
    return destination
