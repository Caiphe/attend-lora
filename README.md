# attend-lora

Training LoRA adapters on top of FLUX.1 diffusion models for stadium/venue image generation.

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url> attend-lora
cd attend-lora
pip install -r requirements.txt
```

Copy the example environment file and fill in your tokens:

```bash
cp .env.example .env
```

## HuggingFace Setup

1. **Get a HuggingFace token.** Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is sufficient). Paste it into `.env` as `HF_TOKEN`.

2. **Accept the FLUX.1-dev license.** Visit the [black-forest-labs/FLUX.1-dev model page](https://huggingface.co/black-forest-labs/FLUX.1-dev) while signed in and click "Agree and access repository." Downloads will fail until you accept.

3. **Verify your setup** by running:

   ```bash
   python scripts/setup_check.py
   ```

## License & Usage

This project trains LoRA adapters on **FLUX.1-dev**, which is released under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). **Non-commercial / research use only.** Do not deploy outputs from this project in any commercial product without obtaining a separate commercial license from Black Forest Labs.

## GPU Requirements

| Tier        | GPU                | VRAM   | Notes                                        |
|-------------|--------------------|--------|----------------------------------------------|
| Minimum     | RTX 3090 / RTX 4090| 24 GB  | Requires 4-bit quantization + small batches  |
| Recommended | A100               | 40 GB  | Comfortable training with mixed precision    |
| Ideal       | A100 / H100        | 80 GB  | Larger batches, faster iteration             |

CPU-only and Apple Silicon are not supported for training.

## Project Layout

```
attend-lora/
  data/
    raw/          # user uploads images here
    processed/    # cleaned + captioned images land here
  outputs/
    loras/        # trained .safetensors adapters saved here
    samples/      # generated test images
  scripts/        # all Python scripts
  requirements.txt
  .env.example
  README.md
```
