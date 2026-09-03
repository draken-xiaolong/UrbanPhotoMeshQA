#!/usr/bin/env python3
"""Rewrite known machine-specific data roots as portable data-root-relative paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REMOTE_ROOT = "/root/autodl-tmp/UrbanPhotoMeshQA/data/"
LOCAL_ROOT = "/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/"


def convert(value):
    if isinstance(value, dict):
        return {key: convert(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert(item) for item in value]
    if isinstance(value, str):
        for root in (REMOTE_ROOT, LOCAL_ROOT):
            if value.startswith(root):
                return value[len(root):]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = convert(json.loads(args.input.read_text(encoding="utf-8")))
    payload["path_protocol"] = {
        "type": "data-root-relative",
        "expected_data_root": "/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
