"""Core inference for FLUX.1-dev + per-stadium LoRA.

Loads the base FluxPipeline once into a module-level singleton. LoRA
weights are attached/swapped on demand: each call ensures the requested
stadium's LoRA is the active adapter, swapping it in only when the
previously loaded one differs. Designed to be imported by
scripts/api.py — not a standalone CLI.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LORAS_DIR = PROJECT_ROOT / "outputs" / "loras"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_TRIGGER_WORD = "sov"

PROMPT_TEMPLATE = (
    "a {trigger} photo of {stadium} stadium, "
    "section {section} row {row} seat {seat}, "
    "{event_type} event, photorealistic wide angle stadium view"
)


@dataclass
class GenerationRequest:
    stadium: str
    section: str
    row: str
    seat: str
    event_type: str = "basketball"
    steps: int = 20
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    trigger_word: str = DEFAULT_TRIGGER_WORD


def build_prompt(req: GenerationRequest) -> str:
    return PROMPT_TEMPLATE.format(
        trigger=req.trigger_word,
        stadium=req.stadium,
        section=req.section,
        row=req.row,
        seat=req.seat,
        event_type=req.event_type,
    )


def lora_path_for(stadium: str) -> Path:
    return LORAS_DIR / f"{stadium}_v1.safetensors"


def list_available_stadiums() -> list[str]:
    if not LORAS_DIR.exists():
        return []
    stadiums: set[str] = set()
    for path in LORAS_DIR.glob("*.safetensors"):
        stem = path.stem
        base, sep, suffix = stem.rpartition("_v")
        if sep and base and suffix.isdigit():
            stadiums.add(base)
        else:
            stadiums.add(stem)
    return sorted(stadiums)


class Generator:
    def __init__(self) -> None:
        self.pipeline = None
        self.base_model: str = DEFAULT_BASE_MODEL
        self.current_lora: Optional[str] = None
        self._lock = threading.Lock()

    def load_pipeline(self, base_model: str = DEFAULT_BASE_MODEL) -> None:
        if self.pipeline is not None and self.base_model == base_model:
            return

        import torch
        from diffusers import FluxPipeline

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. FLUX inference needs an NVIDIA GPU."
            )

        if load_dotenv is not None:
            env_path = PROJECT_ROOT / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        token = os.environ.get("HF_TOKEN") or None

        print(f"Loading {base_model} (this may take several minutes)...")
        pipe = FluxPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            token=token,
        ).to("cuda")
        self.pipeline = pipe
        self.base_model = base_model
        self.current_lora = None
        print("Base pipeline ready.")

    def ensure_lora(self, lora_path: Path) -> None:
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        if self.pipeline is None:
            raise RuntimeError("Call load_pipeline() before ensure_lora().")

        resolved = str(lora_path.resolve())
        if self.current_lora == resolved:
            return

        if self.current_lora is not None:
            self.pipeline.unload_lora_weights()
            self.current_lora = None
        self.pipeline.load_lora_weights(resolved)
        self.current_lora = resolved

    def generate(self, req: GenerationRequest) -> tuple:
        import torch

        with self._lock:
            self.load_pipeline()
            self.ensure_lora(lora_path_for(req.stadium))

            prompt = build_prompt(req)
            torch_generator = None
            if req.seed is not None:
                torch_generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

            start = time.time()
            image = self.pipeline(
                prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=torch_generator,
            ).images[0]
            elapsed = time.time() - start
            return image, prompt, elapsed


generator = Generator()
