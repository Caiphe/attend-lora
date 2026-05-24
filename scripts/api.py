"""FastAPI server exposing FLUX + per-stadium LoRA generation.

Run from the project root:

    uvicorn scripts.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scripts.generate import (
    GenerationRequest,
    generator,
    list_available_stadiums,
    lora_path_for,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "stadiums.json"


def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []
    try:
        data = json.loads(CATALOG_PATH.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def build_stadium_response() -> list[dict]:
    trained = set(list_available_stadiums())
    catalog = load_catalog()
    if catalog:
        out = []
        for entry in catalog:
            slug = entry.get("slug")
            if not slug:
                continue
            out.append({
                "slug": slug,
                "name": entry.get("name") or slug,
                "city": entry.get("city"),
                "capacity": entry.get("capacity"),
                "trained": slug in trained,
            })
        # Surface any trained stadium that isn't in the catalog (custom slugs).
        catalog_slugs = {e["slug"] for e in out}
        for slug in sorted(trained - catalog_slugs):
            out.append({"slug": slug, "name": slug, "city": None, "capacity": None, "trained": True})
        return out
    return [
        {"slug": slug, "name": slug, "city": None, "capacity": None, "trained": True}
        for slug in sorted(trained)
    ]


CURL_EXAMPLE_URL = (
    "curl 'http://localhost:8000/generate/url?"
    "stadium=madison_square_garden&section=122&row=G&seat=14&event=basketball'"
    " --output sample.jpg"
)
CURL_EXAMPLE_POST = (
    "curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' "
    "-d '{\"stadium\":\"madison_square_garden\",\"section\":\"122\","
    "\"row\":\"G\",\"seat\":\"14\",\"event_type\":\"basketball\"}'"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stadiums = list_available_stadiums()
    print()
    print("attend-lora API is ready.")
    print("Endpoints: GET /health  GET /stadiums  POST /generate  GET /generate/url")
    print(f"Available stadiums: {stadiums or '(none — train a LoRA first)'}")
    print()
    print("Test with:")
    print(f"  {CURL_EXAMPLE_URL}")
    print(f"  {CURL_EXAMPLE_POST}")
    print()
    yield


app = FastAPI(title="attend-lora", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateBody(BaseModel):
    stadium: str
    section: str
    row: str
    seat: str
    event_type: str = "basketball"
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = 3.5
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_base64: str
    prompt_used: str
    generation_time_seconds: float
    stadium: str
    section: str


def _to_request(body: GenerateBody) -> GenerationRequest:
    return GenerationRequest(
        stadium=body.stadium,
        section=body.section,
        row=body.row,
        seat=body.seat,
        event_type=body.event_type,
        steps=body.steps,
        guidance_scale=body.guidance_scale,
        width=body.width,
        height=body.height,
        seed=body.seed,
    )


def _generate_or_raise(req: GenerationRequest):
    if not lora_path_for(req.stadium).exists():
        raise HTTPException(status_code=404, detail=f"No LoRA found for stadium '{req.stadium}'")
    try:
        return generator.generate(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "loaded_stadiums": list_available_stadiums()}


@app.get("/stadiums")
def stadiums():
    return {"stadiums": build_stadium_response()}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(body: GenerateBody):
    req = _to_request(body)
    image, prompt, elapsed = _generate_or_raise(req)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    return GenerateResponse(
        image_base64=encoded,
        prompt_used=prompt,
        generation_time_seconds=round(elapsed, 2),
        stadium=req.stadium,
        section=req.section,
    )


@app.get("/generate/url")
def generate_url(
    stadium: str = Query(...),
    section: str = Query(...),
    row: str = Query(...),
    seat: str = Query(...),
    event: str = Query("basketball"),
    steps: int = Query(20, ge=1, le=100),
    guidance_scale: float = Query(3.5),
    width: int = Query(1024, ge=64, le=2048),
    height: int = Query(1024, ge=64, le=2048),
    seed: Optional[int] = Query(None),
):
    req = GenerationRequest(
        stadium=stadium,
        section=section,
        row=row,
        seat=seat,
        event_type=event,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
    )
    image, _prompt, _elapsed = _generate_or_raise(req)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
