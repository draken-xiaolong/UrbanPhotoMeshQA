#!/usr/bin/env python3
"""Export self-contained glTF texture-attack variants and six-view previews."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from urbanphotomeshqa.gltf import GltfReader  # noqa: E402
from urbanphotomeshqa.texture import STANDARD_DIRECTIONS, render_textured_view  # noqa: E402


ATTACKS = {
    "jpeg": (("light", 70), ("medium", 40), ("heavy", 15)),
    "blur": (("light", 0.8), ("medium", 2.0), ("heavy", 4.0)),
    "brightness": (("light", 0.80), ("medium", 0.55), ("heavy", 0.30)),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def degrade(image: Image.Image, attack: str, value: float) -> Image.Image:
    image = image.convert("RGB")
    if attack == "jpeg":
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=int(value), optimize=False)
        stream.seek(0)
        return Image.open(stream).convert("RGB")
    if attack == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if attack == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(value))
    raise ValueError(attack)


def main() -> None:
    args = parse_args()
    source = args.gltf.resolve()
    root = json.loads(source.read_text(encoding="utf-8"))
    buffer_names = [item["uri"] for item in root.get("buffers", []) if item.get("uri")]
    texture_names = [item["uri"] for item in root.get("images", []) if item.get("uri")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rendered_rows = []
    for attack, levels in ATTACKS.items():
        for level, value in levels:
            name = f"{attack}_{level}"
            variant = args.output_dir / attack / level
            texture_dir = variant / "textures"
            preview_dir = variant / "previews"
            texture_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(parents=True, exist_ok=True)
            variant_root = json.loads(json.dumps(root))
            for image_entry in variant_root.get("images", []):
                if image_entry.get("uri"):
                    image_entry["uri"] = f"textures/{Path(image_entry['uri']).name}"
            (variant / source.name).write_text(
                json.dumps(variant_root, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for buffer_name in buffer_names:
                shutil.copy2(source.parent / buffer_name, variant / buffer_name)
            for texture_name in texture_names:
                source_texture = source.parent / texture_name
                target_texture = texture_dir / Path(texture_name).name
                attacked = degrade(Image.open(source_texture), attack, value)
                if target_texture.suffix.lower() in {".jpg", ".jpeg"}:
                    attacked.save(target_texture, quality=95)
                else:
                    attacked.save(target_texture)

            mesh = GltfReader(variant / source.name).load_mesh(include_texture=True)
            previews = []
            row_images = []
            for index, direction in enumerate(STANDARD_DIRECTIONS):
                image = Image.fromarray(
                    render_textured_view(mesh, direction=direction, size=args.render_size)
                )
                preview = preview_dir / f"view{index}.png"
                image.save(preview)
                previews.append(str(preview.relative_to(args.output_dir)))
                row_images.append(image)
            rendered_rows.append((name, row_images))
            records.append({
                "attack": attack,
                "level": level,
                "value": value,
                "gltf": str((variant / source.name).relative_to(args.output_dir)),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "previews": previews,
            })

    label_height = 24
    sheet = Image.new(
        "RGB",
        (len(STANDARD_DIRECTIONS) * args.render_size,
         len(rendered_rows) * (args.render_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, (name, images) in enumerate(rendered_rows):
        top = row * (args.render_size + label_height)
        draw.text((5, top + 4), name, fill="black")
        for column, image in enumerate(images):
            sheet.paste(image, (column * args.render_size, top + label_height))
    sheet.save(args.output_dir / "texture_attack_contact_sheet.png")
    manifest = {
        "source": str(source),
        "seed": args.seed,
        "geometry_unchanged": True,
        "records": records,
    }
    (args.output_dir / "texture_attack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "variants": len(records)}, indent=2))


if __name__ == "__main__":
    main()
