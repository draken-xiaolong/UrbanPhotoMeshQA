#!/usr/bin/env python3
"""Split the fixed source manifest by locked protocol split for parallel export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-chunks", type=int, default=1)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "blind"):
        payload = {**source, "records": [row for row in source["records"] if row["split"] == split]}
        path = args.output_dir / f"source_{split}_seed2026.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(split, len(payload["records"]), path)
        if split == "train" and args.train_chunks > 1:
            for chunk in range(args.train_chunks):
                chunk_payload = {**source, "records": payload["records"][chunk::args.train_chunks]}
                chunk_path = args.output_dir / f"source_train_chunk{chunk + 1}_seed2026.json"
                chunk_path.write_text(json.dumps(chunk_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"train_chunk{chunk + 1}", len(chunk_payload["records"]), chunk_path)


if __name__ == "__main__":
    main()
