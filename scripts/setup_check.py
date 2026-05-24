"""Environment verification script for attend-lora.

Run from the project root:

    python scripts/setup_check.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

GREEN_CHECK = "\033[92m✓\033[0m"
RED_X = "\033[91m✗\033[0m"
YELLOW_WARN = "\033[93m!\033[0m"

MIN_PYTHON = (3, 10)
MIN_VRAM_GB = 20.0

REQUIRED_IMPORTS = [
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "PIL",
    "datasets",
    "huggingface_hub",
    "dotenv",
    "tqdm",
]


def ok(msg: str) -> None:
    print(f"  {GREEN_CHECK} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED_X} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW_WARN} {msg}")


def check_python() -> bool:
    version = sys.version_info
    label = f"Python {version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) >= MIN_PYTHON:
        ok(f"{label} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
        return True
    fail(f"{label} — need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    return False


def check_cuda() -> bool:
    try:
        import torch
    except ImportError:
        fail("CUDA check skipped — torch not installed")
        return False

    if not torch.cuda.is_available():
        fail("CUDA not available — training requires an NVIDIA GPU")
        return False

    device_name = torch.cuda.get_device_name(0)
    ok(f"CUDA available — {device_name}")
    return True


def check_vram() -> bool:
    try:
        import torch
    except ImportError:
        fail("VRAM check skipped — torch not installed")
        return False

    if not torch.cuda.is_available():
        fail("VRAM check skipped — no CUDA device")
        return False

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    label = f"GPU VRAM {vram_gb:.1f} GB"
    if vram_gb >= MIN_VRAM_GB:
        ok(f"{label} (>= {MIN_VRAM_GB:.0f} GB)")
        return True
    warn(f"{label} — recommended >= {MIN_VRAM_GB:.0f} GB; training may OOM")
    return False


def check_hf_token() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        fail("HF_TOKEN check skipped — python-dotenv not installed")
        return False

    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        warn(f"No .env file at {env_path} — falling back to process environment")

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token == "your_huggingface_token_here":
        fail("HF_TOKEN is not set in .env")
        return False
    ok("HF_TOKEN is set")
    return True


def check_imports() -> bool:
    all_ok = True
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            ok(f"import {module_name}")
        except ImportError as exc:
            fail(f"import {module_name} — {exc}")
            all_ok = False
    return all_ok


def main() -> int:
    print("attend-lora setup check\n")

    checks = [
        ("Python version", check_python),
        ("CUDA", check_cuda),
        ("GPU VRAM", check_vram),
        ("HF_TOKEN", check_hf_token),
        ("Required imports", check_imports),
    ]

    results: list[bool] = []
    for label, fn in checks:
        print(f"[{label}]")
        results.append(fn())
        print()

    if all(results):
        print(f"{GREEN_CHECK} All checks passed.")
        return 0
    print(f"{RED_X} Some checks failed — review the output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
