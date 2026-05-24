"""Read a YAML config and invoke train_lora.py with the right arguments.

Usage from project root:

    python scripts/train_from_config.py configs/example_stadium.yaml [extra train_lora.py args...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_lora.py"

CONFIG_TO_CLI = {
    "stadium_name": "--stadium_name",
    "data_dir": "--data_dir",
    "captions_file": "--captions_file",
    "output_dir": "--output_dir",
    "base_model": "--base_model",
    "steps": "--steps",
    "learning_rate": "--learning_rate",
    "lora_rank": "--lora_rank",
    "lora_alpha": "--lora_alpha",
    "batch_size": "--batch_size",
    "resolution": "--resolution",
    "mixed_precision": "--mixed_precision",
    "save_every_n_steps": "--save_every_n_steps",
    "trigger_word": "--trigger_word",
    "seed": "--seed",
    "num_workers": "--num_workers",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to train_lora.py (override config values)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return 1

    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        print(f"ERROR: config must be a YAML mapping, got {type(config).__name__}")
        return 1

    cmd: list[str] = [sys.executable, str(TRAIN_SCRIPT)]
    unknown: list[str] = []
    for key, value in config.items():
        flag = CONFIG_TO_CLI.get(key)
        if flag is None:
            unknown.append(key)
            continue
        cmd.extend([flag, str(value)])

    if config.get("gradient_checkpointing") is False:
        cmd.append("--no-gradient_checkpointing")
    elif config.get("gradient_checkpointing") is True:
        cmd.append("--gradient_checkpointing")

    cmd.extend(args.extra)

    if unknown:
        print(f"Note: ignoring unknown config keys: {', '.join(unknown)}")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
