# Training attend-lora on a Rented Cloud GPU (RunPod)

You can't train FLUX LoRA on this Mac. This doc walks you through renting a single A100, syncing the project, and running the training pipeline end-to-end.

**Expected cost:** ~$3-8 for one 1500-step training run on an A100 40GB (~2-4 hours wall-clock).

## 1. Pick a provider

| Provider     | $/hr (A100 40GB) | Notes |
|--------------|------------------|-------|
| **RunPod**   | $1.89 on-demand / ~$1.20 spot | What this doc assumes. Good balance of price + UX. |
| Vast.ai      | $0.60-1.50 spot | Cheaper but instances vanish more often. |
| Lambda Labs  | $1.29 on-demand | Cleaner UX, less GPU selection. |
| Modal        | ~$3/hr (serverless) | No pod management, but expensive for long runs. |

If A100 40GB is unavailable, fallback order: **A6000 48GB → RTX 4090 24GB**. The 4090 works with `--gradient_checkpointing` + `--mixed_precision bf16`, but expect tighter memory.

## 2. RunPod account + SSH key

1. Sign up at [runpod.io](https://runpod.io) and add credit (~$20 is plenty for several runs).
2. Generate an SSH key locally if you don't have one:
   ```bash
   ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "you@example.com"
   ```
3. In RunPod: **Settings → SSH Public Keys** → paste `~/.ssh/id_ed25519.pub`.

## 3. Create a persistent network volume (one-time)

This saves the FLUX.1-dev model (~24 GB) and BLIP-2 (~10 GB) between sessions so you don't redownload every time you spin up a pod.

1. **Storage → Network Volumes → New Network Volume**
2. Datacenter: pick one with A100 availability (e.g. `CA-MTL-1` or `US-OR-1`).
3. Size: **100 GB** (room for the HF cache + your dataset + outputs).
4. Cost: ~$7/month while it exists. Delete it when you're done with the project.

## 4. Spin up the pod

1. **Pods → Deploy → GPU Pod**
2. GPU: **1× A100 40GB PCIe** (Spot is cheaper if you don't mind possible interruption).
3. Template: **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`** (or the latest equivalent).
4. **Edit Template** before deploy:
   - Container Disk: `40 GB`
   - Volume Mount Path: `/workspace` (default)
   - Attach your network volume to `/workspace`
   - Expose TCP Ports: `22` (SSH) and `8000` (for the API if you want to test it from the pod).
5. **Deploy On-Demand** (or Spot).

Once the pod is "Running", copy the SSH command from the **Connect** dropdown — it looks like:
```
ssh root@123.456.78.90 -p 22 -i ~/.ssh/id_ed25519
```

## 5. Sync the project up from your Mac

From `~/Dev/AI/attend-lora` on your Mac, push code + data using `rsync`. Exclude the local venv and large outputs:

```bash
# Replace HOST/PORT with values from the RunPod Connect tab.
HOST=root@123.456.78.90
PORT=22

rsync -avz -e "ssh -p $PORT" \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'outputs/' \
  ./ "$HOST":/workspace/attend-lora/
```

`outputs/` is excluded because you'll generate fresh outputs on the pod and pull them back. `data/raw/` and `data/processed/` *are* included — adjust the excludes if your raw image set is huge and you'd rather upload it separately.

## 6. Bootstrap the pod (run these on the pod, not locally)

```bash
ssh -p $PORT $HOST
cd /workspace/attend-lora

# Point the HuggingFace cache at the network volume so re-rentals reuse it.
export HF_HOME=/workspace/hf-cache
echo 'export HF_HOME=/workspace/hf-cache' >> ~/.bashrc

# Fresh venv on the pod's Python.
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Sanity check.
python scripts/setup_check.py
```

Expected `setup_check.py` output on a healthy pod: Python ✓, CUDA ✓, VRAM 40+ GB ✓, all imports ✓ (including `bitsandbytes` and `xformers`, which both work on Linux + NVIDIA).

## 7. Configure HuggingFace access

```bash
cp .env.example .env
nano .env   # paste your HF_TOKEN from huggingface.co/settings/tokens
```

**Critical:** open [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) in a browser while logged in and click **Agree and access repository**. The pod's download will 403 until you do this.

## 8. Run the training pipeline

```bash
# 1. Prepare images: resize, dedupe-by-quality, build metadata
python scripts/prepare_data.py --event-type basketball

# 2. Caption with BLIP-2 (downloads ~10GB the first time)
python scripts/caption_images.py

# 3. Train the LoRA (downloads ~24GB FLUX weights on first run)
python scripts/train_from_config.py configs/example_stadium.yaml
```

Watch the loss in the tqdm bar. With 50 training images and the defaults, expect:
- VAE + text-encoder load: ~3 min
- FLUX weights download (first time only): ~15 min
- Training: ~2-4 hours for 1500 steps on A100 40GB
- Sample images saved every 500 steps to `outputs/samples/<stadium>/step_N/`

## 9. Pull the trained LoRA back to your Mac

When training prints `LoRA saved to: outputs/loras/madison_square_garden_v1.safetensors`:

```bash
# Run this from your Mac, not the pod.
rsync -avz -e "ssh -p $PORT" \
  "$HOST":/workspace/attend-lora/outputs/ \
  ./outputs/
```

You can now load the LoRA in your local `scripts/api.py` *for endpoint testing* (though actually generating images still needs the pod — see next section).

## 10. Optional: run the API on the pod for live inference

Inference also needs the GPU, so spin the API up on the pod and tunnel:

```bash
# On the pod:
uvicorn scripts.api:app --host 0.0.0.0 --port 8000

# On your Mac (separate terminal):
ssh -p $PORT -L 8000:localhost:8000 $HOST

# Now open http://localhost:8000/health on your Mac.
```

Or use the pod's exposed port 8000 directly via the public URL RunPod gives you.

## 11. Stop the pod when done

**RunPod → Pods → ⋯ → Stop** (preserves the pod state, billed for storage only) or **Terminate** (kills it, billed only for the network volume).

If you're done with the project entirely, also delete the network volume from **Storage**.

## Common pitfalls

| Symptom | Fix |
|---|---|
| `403 forbidden` downloading FLUX | Accept the license at the model page (step 7). |
| `CUDA OOM` during training | Add `--resolution 768` or `--lora_rank 8`; ensure `--gradient_checkpointing` is on. |
| `bitsandbytes` won't import | You're on a non-CUDA pod. Pick a CUDA-enabled template. |
| Pod terminates mid-training (spot) | The `--save_every_n_steps` checkpoints in `outputs/loras/` let you continue from the last one. Resume support isn't built in yet — for now, re-run with `--steps` set to the remainder. |
| First request to the API hangs for minutes | Expected — `generate.py` lazy-loads FluxPipeline on first call. Subsequent requests are fast. |
| HF cache re-downloads every pod | You didn't mount the network volume at `/workspace`, or didn't set `HF_HOME=/workspace/hf-cache`. |

## Cost-optimization checklist

- Use **spot pricing** for training runs — interruption is rare and you save ~40%.
- Use a **network volume** so you only pay the FLUX download cost once.
- **Stop the pod** (don't just close the SSH session) when not actively training.
- For repeated training runs, leave the network volume and re-create pods on demand. The volume is ~$0.07/GB/month — 100 GB ≈ $7/month idle.
