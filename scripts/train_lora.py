"""Train a LoRA adapter on FLUX.1-dev for stadium seat-view generation.

DreamBooth-style single-GPU training using diffusers + peft. Trains LoRA
on the FLUX transformer with rectified-flow loss. Saves the final adapter
to outputs/loras/{stadium_name}_v1.safetensors.

Usage from project root:

    python scripts/train_lora.py --stadium_name madison_square_garden
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "to_add_q", "to_add_k", "to_add_v", "to_add_out",
]

A100_USD_PER_STEP = 0.000006


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stadium_name", required=True)
    p.add_argument("--data_dir", default=str(PROJECT_ROOT / "data" / "processed" / "images"))
    p.add_argument("--captions_file", default=str(PROJECT_ROOT / "data" / "processed" / "training_captions.txt"))
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "loras"))
    p.add_argument("--base_model", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_every_n_steps", type=int, default=500)
    p.add_argument("--trigger_word", default="sov")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def load_captions(captions_file: Path) -> dict[str, str]:
    if not captions_file.exists():
        return {}
    captions: dict[str, str] = {}
    for line in captions_file.read_text().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        fname, caption = line.split("|", 1)
        captions[fname.strip()] = caption.strip()
    return captions


class StadiumDataset(Dataset):
    def __init__(self, data_dir: Path, captions: dict[str, str], instance_prompt: str, resolution: int):
        self.entries: list[tuple[Path, str]] = []
        for path in sorted(data_dir.glob("*.jpg")):
            self.entries.append((path, captions.get(path.name) or instance_prompt))
        if not self.entries:
            raise RuntimeError(f"No training images found in {data_dir}")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        path, caption = self.entries[idx]
        img = Image.open(path).convert("RGB")
        return {"pixel_values": self.transform(img), "prompt": caption}


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
    }


def cycle(loader: DataLoader):
    while True:
        for b in loader:
            yield b


def save_lora(transformer, output_dir: Path, filename: str) -> Path:
    from diffusers import FluxPipeline
    from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(transformer))
    output_dir.mkdir(parents=True, exist_ok=True)
    FluxPipeline.save_lora_weights(
        save_directory=str(output_dir),
        transformer_lora_layers=state,
        weight_name=filename,
    )
    return output_dir / filename


def generate_samples(pipeline, stadium_name: str, trigger_word: str, step: int, samples_root: Path, seed: int) -> None:
    step_dir = samples_root / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        f"a {trigger_word} photo of {stadium_name} stadium, section 100, courtside view",
        f"a {trigger_word} photo of {stadium_name} stadium, upper deck, behind the basket",
    ]
    pipeline.transformer.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            gen = torch.Generator(device=pipeline.device).manual_seed(seed + i)
            img = pipeline(
                prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=gen,
            ).images[0]
            img.save(step_dir / f"sample_{i}.png")


def main() -> int:
    args = parse_args()

    if load_dotenv is not None:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    hf_token = os.environ.get("HF_TOKEN") or None

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. FLUX LoRA training requires an NVIDIA GPU "
              "(see README — minimum RTX 3090 24GB, recommended A100 40GB).")
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map[args.mixed_precision]

    from diffusers import FluxPipeline
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    print(f"Loading {args.base_model} (this may take several minutes)...")
    pipeline = FluxPipeline.from_pretrained(
        args.base_model,
        torch_dtype=weight_dtype,
        token=hf_token,
    )
    pipeline.to(device)

    transformer = pipeline.transformer
    vae = pipeline.vae

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES,
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable LoRA params: {trainable_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)
    warmup_steps = max(1, args.steps // 10)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.steps,
    )

    instance_prompt = f"a {args.trigger_word} photo of {args.stadium_name} stadium seat view"
    captions = load_captions(Path(args.captions_file))
    dataset = StadiumDataset(Path(args.data_dir), captions, instance_prompt, args.resolution)
    print(f"Dataset: {len(dataset)} images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    batch_iter = cycle(loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_root = PROJECT_ROOT / "outputs" / "samples" / args.stadium_name
    samples_root.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    pbar = tqdm(range(1, args.steps + 1), desc="Training", unit="step")
    start_time = time.time()

    try:
        for step in pbar:
            transformer.train()
            batch = next(batch_iter)
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            prompts = batch["prompt"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompts,
                    device=device,
                    max_sequence_length=512,
                )

            b, c, h, w = latents.shape
            packed = FluxPipeline._pack_latents(latents, b, c, h, w)
            img_ids = FluxPipeline._prepare_latent_image_ids(b, h // 2, w // 2, device, weight_dtype)

            noise = torch.randn_like(packed)
            # Logit-normal timestep sampling, standard for FLUX flow-matching.
            t = torch.sigmoid(torch.randn(b, device=device, dtype=weight_dtype))
            t_b = t.view(-1, 1, 1)
            noisy = (1.0 - t_b) * packed + t_b * noise
            target = noise - packed

            guidance = None
            if getattr(transformer.config, "guidance_embeds", False):
                guidance = torch.full((b,), 1.0, device=device, dtype=weight_dtype)

            pred = transformer(
                hidden_states=noisy,
                timestep=t,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
            if step % 50 == 0:
                avg = sum(losses[-50:]) / min(50, len(losses))
                tqdm.write(f"  step {step}: loss(50)={avg:.4f}  lr={lr_scheduler.get_last_lr()[0]:.2e}")

            if step % args.save_every_n_steps == 0 and step < args.steps:
                save_lora(transformer, output_dir, f"{args.stadium_name}_step_{step}.safetensors")
                generate_samples(pipeline, args.stadium_name, args.trigger_word, step, samples_root, args.seed)

    except torch.cuda.OutOfMemoryError:
        print()
        print("CUDA OUT OF MEMORY")
        print("Try one or more of:")
        print(f"  --batch_size 1                       (current: {args.batch_size})")
        print(f"  --resolution 768  or  --resolution 512   (current: {args.resolution})")
        if not args.gradient_checkpointing:
            print("  --gradient_checkpointing             (currently off)")
        print(f"  --mixed_precision bf16               (current: {args.mixed_precision})")
        print("  reduce --lora_rank (e.g. 8 or 4)")
        print("  offload text encoders to CPU between encode_prompt calls (advanced)")
        return 2

    elapsed = time.time() - start_time
    final_loss = sum(losses[-50:]) / min(50, len(losses)) if losses else float("nan")
    final_filename = f"{args.stadium_name}_v1.safetensors"
    final_path = save_lora(transformer, output_dir, final_filename)
    cost = args.steps * A100_USD_PER_STEP

    print()
    print("Training complete")
    print(f"  Steps:                    {args.steps}")
    print(f"  Final loss (50-step avg): {final_loss:.4f}")
    print(f"  Wall time:                {elapsed / 60:.1f} min")
    print(f"  Estimated cost:           ${cost:.4f}  (A100 @ ${A100_USD_PER_STEP}/step)")
    print(f"  LoRA saved to:            {final_path}")
    print()
    print("Inference:")
    print(f"  from diffusers import FluxPipeline")
    print(f"  import torch")
    print(f"  pipe = FluxPipeline.from_pretrained('{args.base_model}', torch_dtype=torch.bfloat16).to('cuda')")
    print(f"  pipe.load_lora_weights('{final_path}')")
    print(f"  prompt = 'a {args.trigger_word} photo of {args.stadium_name} stadium, section 100, courtside view'")
    print(f"  img = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]")
    print(f"  img.save('sample.png')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
