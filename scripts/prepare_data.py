"""Prepare raw stadium images for LoRA training.

Walks data/raw/, applies quality filters, resizes survivors to 1024x1024
(center-cropped) into data/processed/images/, and writes metadata to
data/processed/metadata.json.

Usage from project root:

    python scripts/prepare_data.py [--event-type basketball]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_DIMENSION = 512
MIN_BLUR_VARIANCE = 100.0
OUTPUT_SIZE = 1024

FILENAME_RE = re.compile(
    r"^section_(?P<section>\d+)_row_(?P<row>[A-Za-z]+)_seat_(?P<seat>\d+)$",
    re.IGNORECASE,
)

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger("prepare_data")


@dataclass
class SkipCounts:
    too_small: int = 0
    too_blurry: int = 0
    corrupted: int = 0

    @property
    def total(self) -> int:
        return self.too_small + self.too_blurry + self.corrupted


def laplacian_variance(img: Image.Image) -> float:
    """Variance of the 4-neighbor Laplacian on the grayscale image."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def center_crop_resize(img: Image.Image, size: int = OUTPUT_SIZE) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((size, size), Image.LANCZOS)


def parse_filename(stem: str) -> dict[str, Optional[str]]:
    match = FILENAME_RE.match(stem)
    if not match:
        return {"section": None, "row": None, "seat": None}
    return {
        "section": match.group("section"),
        "row": match.group("row").upper(),
        "seat": match.group("seat"),
    }


def load_env() -> None:
    if load_dotenv is None:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def discover_images(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-type",
        default="basketball",
        help="Event type assigned to every metadata entry (default: basketball)",
    )
    args = parser.parse_args()

    load_env()
    stadium = os.environ.get("STADIUM_NAME", "unknown_stadium")

    if not RAW_DIR.exists():
        log.error(f"Raw directory not found: {RAW_DIR}")
        return 1

    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    images = discover_images(RAW_DIR)
    if not images:
        log.warning(f"No images found in {RAW_DIR}")
        METADATA_PATH.write_text("[]\n")
        return 0

    skipped = SkipCounts()
    metadata: list[dict] = []

    for src in tqdm(images, desc="Processing", unit="img"):
        try:
            with Image.open(src) as img:
                img.load()
                width, height = img.size
                if width < MIN_DIMENSION or height < MIN_DIMENSION:
                    skipped.too_small += 1
                    continue
                if laplacian_variance(img) < MIN_BLUR_VARIANCE:
                    skipped.too_blurry += 1
                    continue
                processed = center_crop_resize(img)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped.corrupted += 1
            log.warning(f"  corrupted: {src.name} ({exc})")
            continue

        out_name = f"{src.stem}.jpg"
        out_path = PROCESSED_IMAGES_DIR / out_name
        processed.save(out_path, format="JPEG", quality=95)

        parts = parse_filename(src.stem)
        metadata.append({
            "file": out_name,
            "stadium": stadium,
            "section": parts["section"],
            "row": parts["row"],
            "seat": parts["seat"],
            "event_type": args.event_type,
            "camera_fov": None,
            "elevation": None,
            "azimuth": None,
            "caption": "",
        })

    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print()
    print("Summary")
    print(f"  Found:    {len(images)}")
    print(f"  Passed:   {len(metadata)}")
    print(f"  Skipped:  {skipped.total}")
    print(f"    too small (<{MIN_DIMENSION}x{MIN_DIMENSION}):       {skipped.too_small}")
    print(f"    too blurry (variance < {MIN_BLUR_VARIANCE:.0f}):    {skipped.too_blurry}")
    print(f"    corrupted / unreadable:           {skipped.corrupted}")
    print(f"  Metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Images:   {PROCESSED_IMAGES_DIR.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
