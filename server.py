"""mlx-embedding-server — single-purpose multimodal embedding HTTP server (MLX).

Runs a heavy mlx-embeddings model in a dedicated process so clients only need a
thin HTTP client. Apple Silicon (MLX) only. Defaults to Qwen3-VL-Embedding-2B.

Usage:
    uv run python server.py

Endpoint:
    POST /embed  {"image_base64": "<png/jpeg base64>", "model_id": "<optional>"}
        -> {"embedding": [...], "model_id": "...", "dimension": N}
    GET  /health -> {"status": "ok", "model_id": "..."}

MLX arrays/streams are thread-affine, so all MLX work (model load and inference)
is funneled through a single dedicated worker thread. FastAPI would otherwise run
sync endpoints in an arbitrary threadpool worker, which breaks MLX's stream
affinity ("There is no Stream(gpu, N) in current thread").
"""

import asyncio
import base64
import io
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from mlx_embeddings import load
from PIL import Image
from pydantic import BaseModel

DEFAULT_MODEL_ID = os.environ.get(
    "MLX_EMBEDDING_MODEL_ID", "mlx-community/Qwen3-VL-Embedding-2B-mxfp8"
)
TEMPLATE_SOURCE_MODEL_ID = os.environ.get(
    "MLX_EMBEDDING_TEMPLATE_SOURCE", "Qwen/Qwen3-VL-Embedding-2B"
)

app = FastAPI(title="mlx-embedding-server")

# Single thread owns every MLX operation (load + inference) for stream affinity.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

_state: dict[str, Any] = {"model_id": None, "model": None, "processor": None}


def _ensure_model(model_id: str) -> None:
    if _state["model_id"] == model_id and _state["model"] is not None:
        return

    model, processor = load(model_id)

    # The MLX-converted processor may be missing the chat template / token-id
    # attributes that Transformers normally sets up. Patch them in.
    if processor.chat_template is None:
        chat_template = processor._load_chat_template(TEMPLATE_SOURCE_MODEL_ID)
        processor.processor.chat_template = chat_template
        processor.tokenizer.chat_template = chat_template

    processor.processor.image_ids = [processor.image_token_id]
    processor.processor.video_ids = [processor.video_token_id]
    processor.processor.audio_ids = []

    _state.update(model_id=model_id, model=model, processor=processor)


def _embed_sync(model_id: str, image: Image.Image) -> list[float]:
    _ensure_model(model_id)
    embeddings = _state["model"].process(
        [{"image": image}], processor=_state["processor"]
    )
    mx.eval(embeddings)
    return embeddings.tolist()[0]


class EmbedRequest(BaseModel):
    image_base64: str
    model_id: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_id": _state["model_id"]}


@app.post("/embed")
async def embed(request: EmbedRequest) -> dict[str, Any]:
    model_id = request.model_id or DEFAULT_MODEL_ID

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    loop = asyncio.get_running_loop()

    try:
        vector = await loop.run_in_executor(
            _executor, _embed_sync, model_id, image
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"embed failed: {exc}")

    return {
        "embedding": vector,
        "model_id": model_id,
        "dimension": len(vector),
    }


if __name__ == "__main__":
    host = os.environ.get("MLX_EMBEDDING_HOST", "0.0.0.0")
    port = int(os.environ.get("MLX_EMBEDDING_PORT", "8900"))

    if os.environ.get("MLX_EMBEDDING_PRELOAD", "1") == "1":
        _executor.submit(_ensure_model, DEFAULT_MODEL_ID).result()

    uvicorn.run(app, host=host, port=port)
