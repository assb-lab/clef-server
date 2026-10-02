"""Clef (Cloudflare/clef, Cloudflare/clef-flash) inference server exposing the Jev/SystemOne API."""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import snapshot_download
from PIL import Image

MODELS = {"clef": "Cloudflare/clef", "clef-flash": "Cloudflare/clef-flash"}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

logger = logging.getLogger("clef-server")


@dataclass
class Config:
    model: str = "clef"
    revision: str | None = None
    device: str | None = None
    dtype: str = "bfloat16"
    max_length: int = 16384

    @property
    def model_id(self) -> str:
        """Resolve an alias (clef, clef-flash) to a repo ID; anything else is a repo ID or local path."""
        return MODELS.get(self.model, self.model)

    @property
    def model_name(self) -> str:
        return self.model_id.rstrip("/").rsplit("/", 1)[-1]


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_image(source: str) -> Image.Image:
    """Accept an http(s) URL, a data URI, or a bare base64 string."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as response:
            data = response.read()
    else:
        if source.startswith("data:"):
            source = source.split(",", 1)[1]
        data = base64.b64decode(source)
    return Image.open(io.BytesIO(data)).convert("RGB")


def prepare_request(body: dict[str, Any]) -> dict[str, Any]:
    request = dict(body)
    try:
        if images := request.get("images"):
            request["images"] = [decode_image(image) for image in images]
        if videos := request.get("videos"):
            # Each video is a list of frames; frames become a (T, H, W, 3) array.
            request["videos"] = [np.stack([np.asarray(decode_image(frame)) for frame in video]) for video in videos]
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"invalid media: {error}") from error
    return request


class ClefRuntime:
    """One model instance on one device. Forward passes are serialized."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device: str | None = None
        self.model: Any = None
        self.processor: Any = None
        self._systemone: Any = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        path = snapshot_download(self.config.model_id, revision=self.config.revision)
        # joint_schema_model.py ships with the model repo (custom code).
        sys.path.insert(0, path)
        from joint_schema_model import load_release_model, systemone

        self.device = pick_device(self.config.device)
        logger.info("loading %s on %s (%s)", self.config.model_id, self.device, self.config.dtype)
        started = time.perf_counter()
        self.model, self.processor = load_release_model(path, device=self.device, dtype=DTYPES[self.config.dtype])
        self._systemone = systemone
        logger.info("loaded in %.1fs", time.perf_counter() - started)

    def systemone(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._systemone(self.model, self.processor, request, max_length=self.config.max_length)


def create_app(config: Config, runtime: ClefRuntime | None = None) -> FastAPI:
    runtime = runtime or ClefRuntime(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not runtime.ready:
            runtime.load()
        yield

    app = FastAPI(title="clef-server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if runtime.ready else "loading",
            "model": config.model_id,
            "device": runtime.device,
        }

    @app.post("/v1/systemone")
    async def v1_systemone(body: dict[str, Any]) -> dict[str, Any]:
        body.setdefault("model", config.model_name)
        request = await run_in_threadpool(prepare_request, body)
        started = time.perf_counter()
        try:
            response = await run_in_threadpool(runtime.systemone, request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        response["usage"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="clef-server", description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=env("CLEF_MODEL", "clef"),
        help="clef (27B) | clef-flash (9B) | a Hugging Face repo ID or local path (default: clef)",
    )
    parser.add_argument("--revision", default=env("CLEF_REVISION"), help="model revision (commit SHA, branch, tag)")
    parser.add_argument("--device", default=env("CLEF_DEVICE"), help="cuda, cuda:1, mps, cpu (default: auto)")
    parser.add_argument("--dtype", default=env("CLEF_DTYPE", "bfloat16"), choices=sorted(DTYPES))
    parser.add_argument("--max-length", type=int, default=int(env("CLEF_MAX_LENGTH", "16384")))
    parser.add_argument("--host", default=env("HOST", "0.0.0.0"))
    parser.add_argument("--port", "-p", type=int, default=int(env("PORT", "8000")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config(
        model=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
