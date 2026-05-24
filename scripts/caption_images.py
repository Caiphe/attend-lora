"""Generate captions for processed stadium images using BLIP-2.

Reads data/processed/metadata.json, fills in missing captions via BLIP-2
conditional generation (Salesforce/blip2-opt-2.7b), and writes a
diffusers-style training_captions.txt alongside it.

Usage from project root:

    python scripts/caption_images.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
TRAINING_CAPTIONS_PATH = PROCESSED_DIR / "training_captions.txt"

BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
MAX_NEW_TOKENS = 60
READINESS_MIN_IMAGES = 20

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger("caption_images")


def load_env() -> None:
    if load_dotenv is None:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def humanize(name: str | None, fallback: str) -> str:
    if not name:
        return fallback
    cleaned = name.replace("_", " ").strip()
    return cleaned.title() if cleaned else fallback


def build_prompt(entry: dict) -> str:
    section_value = entry.get("section")
    section = f"section {section_value}" if section_value else "the stands"
    stadium = humanize(entry.get("stadium"), "the stadium")
    event_type = entry.get("event_type") or "sporting"
    return (
        f"A photorealistic point-of-view image from {section} at {stadium}, "
        f"showing the view a spectator would see during a {event_type} event."
    )


def load_model():
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        log.warning("CUDA not available — BLIP-2 will run on CPU (very slow).")
        device = "cpu"
        dtype = torch.float32

    log.info(f"Loading {BLIP2_MODEL_ID} on {device} ({dtype})...")
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model, device, dtype


def generate_caption(processor, model, device, dtype, image: Image.Image, prompt: str) -> str:
    import torch

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    if dtype != torch.float32:
        inputs = {
            k: (v.to(dtype) if v.dtype.is_floating_point else v)
            for k, v in inputs.items()
        }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if not decoded:
        return prompt
    if decoded.lower().startswith(prompt.lower()):
        return decoded
    return f"{prompt} {decoded}".strip()


def write_training_file(metadata: list[dict]) -> None:
    lines = [f"{e['file']}|{e['caption']}" for e in metadata if e.get("caption")]
    body = "\n".join(lines)
    TRAINING_CAPTIONS_PATH.write_text(body + ("\n" if body else ""))


def print_summary(metadata: list[dict], captioned: int, failed: int) -> None:
    total = len(metadata)
    with_caption = sum(1 for e in metadata if e.get("caption"))
    print()
    print("Summary")
    print(f"  Total metadata entries: {total}")
    print(f"  Captioned this run:     {captioned}")
    print(f"  Failed this run:        {failed}")
    print(f"  Entries with captions:  {with_caption}")
    print(f"  Training file:          {TRAINING_CAPTIONS_PATH.relative_to(PROJECT_ROOT)}")
    if with_caption >= READINESS_MIN_IMAGES:
        print(f"  Status: READY ({with_caption} >= {READINESS_MIN_IMAGES} minimum)")
    else:
        needed = READINESS_MIN_IMAGES - with_caption
        print(f"  Status: NOT READY — add {needed} more captioned images "
              f"(need >= {READINESS_MIN_IMAGES})")


def main() -> int:
    load_env()

    if not METADATA_PATH.exists():
        log.error(f"Metadata not found: {METADATA_PATH}. Run prepare_data.py first.")
        return 1

    metadata = json.loads(METADATA_PATH.read_text())
    if not metadata:
        log.warning("metadata.json is empty — nothing to caption.")
        write_training_file(metadata)
        print_summary(metadata, captioned=0, failed=0)
        return 0

    pending = [entry for entry in metadata if not entry.get("caption")]
    if not pending:
        log.info("All entries already have captions — rewriting training_captions.txt.")
        write_training_file(metadata)
        print_summary(metadata, captioned=0, failed=0)
        return 0

    try:
        processor, model, device, dtype = load_model()
    except Exception as exc:
        log.error(f"Failed to load BLIP-2: {exc}")
        return 1

    captioned = 0
    failed = 0

    for entry in tqdm(pending, desc="Captioning", unit="img"):
        image_path = PROCESSED_IMAGES_DIR / entry["file"]
        try:
            with Image.open(image_path) as img:
                img.load()
                rgb = img.convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError, OSError) as exc:
            log.warning(f"  skipped {entry['file']} (image error: {exc})")
            failed += 1
            continue

        try:
            prompt = build_prompt(entry)
            entry["caption"] = generate_caption(processor, model, device, dtype, rgb, prompt)
            captioned += 1
        except Exception as exc:
            log.warning(f"  skipped {entry['file']} (caption error: {exc})")
            failed += 1

    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    write_training_file(metadata)
    print_summary(metadata, captioned=captioned, failed=failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
