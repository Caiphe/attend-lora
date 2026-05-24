# attend-lora UI

A Vite + React presentation demo for the Attend AI venue intelligence platform. Talks to the FastAPI server (`scripts/api.py`) at `http://localhost:8000`.

## Setup

```bash
npm install
npm run dev    # serves on http://localhost:5173
```

## Prerequisite — start the API first

The UI is a thin client; the model lives in the FastAPI server. From the project root (one level up from `ui/`):

```bash
source venv/bin/activate
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
```

If the API isn't running, the top-right status pill goes red ("Offline") and the **Generate View** button is disabled.

## What you'll see

- **Stats bar** — live health from `GET /health`, count of stadium LoRAs from `GET /stadiums`, pulsing green dot when the API is healthy.
- **Left panel** — stadium dropdown (auto-fills with whatever LoRAs the server reports), section/row/seat inputs, event type, and an Advanced disclosure for steps / guidance / seed. The exact prompt that'll be POSTed is rendered live at the bottom.
- **Right panel** — idle placeholder → shimmer-skeleton loader with a fake-but-plausible progress bar (0 → 90% over `steps × 1.5s`, jumps to 100% when the server responds) → final image with section pill, generation time, expandable prompt, and download button.

## Configuration

The API base URL is hard-coded in `src/App.jsx`:

```js
const API_BASE = 'http://localhost:8000';
```

Change it there if you tunnel from a remote pod (e.g. `http://localhost:8000` via `ssh -L 8000:localhost:8000`) or proxy via a different host.

## Build

```bash
npm run build    # outputs dist/
npm run preview  # serves the built bundle on 4173
```
