"""Content-addressed integrity helpers for glTF asset packages and feature caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gltf_dependencies(gltf: Path) -> list[Path]:
    root = json.loads(gltf.read_text(encoding="utf-8"))
    paths = [gltf]
    for entry in root.get("buffers", []):
        uri = entry.get("uri")
        if uri and not uri.startswith("data:"):
            paths.append((gltf.parent / uri).resolve())
    for entry in root.get("images", []):
        uri = entry.get("uri")
        if uri and not uri.startswith("data:"):
            paths.append((gltf.parent / uri).resolve())
    return paths


def asset_digest(gltf: Path) -> tuple[str, list[dict[str, object]]]:
    digest = hashlib.sha256()
    records = []
    for path in gltf_dependencies(gltf):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(gltf.parent).as_posix() if path != gltf else gltf.name
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hash})
    return digest.hexdigest(), records


def extractor_signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
