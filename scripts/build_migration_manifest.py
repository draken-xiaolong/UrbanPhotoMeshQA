#!/usr/bin/env python3
"""Build a reproducible old-to-new migration manifest using the freeze hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--freeze-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_hash = defaultdict(list)
    for line in args.freeze_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value, relative = line.split("  ", 1)
        by_hash[value].append(relative.removeprefix("./"))
    records = []
    skip = {args.output.resolve(), args.freeze_list.resolve()}
    for path in sorted(item for item in args.new_root.rglob("*") if item.is_file()):
        if path.resolve() in skip or ".pytest_cache" in path.parts:
            continue
        value = digest(path)
        candidates = by_hash.get(value, [])
        if not candidates:
            continue
        relative = str(path.relative_to(args.new_root))
        same = relative if relative in candidates else candidates[0]
        records.append({
            "old_path": str(args.old_root / same),
            "new_path": str(path),
            "sha256": value,
            "reason": "quality-assessment dependency, pretrained backbone, baseline, or reproducibility artifact",
        })
    payload = {"schema_version": 1, "old_project_frozen": True,
               "old_root": str(args.old_root), "new_root": str(args.new_root),
               "matched_migrated_files": len(records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"matched_migrated_files": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
