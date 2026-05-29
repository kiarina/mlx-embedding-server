# mlx-embedding-server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform: macOS (Apple Silicon)](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-black)
![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

*English | [日本語](README.ja.md)*

A single-purpose HTTP server that serves multimodal embeddings via
[`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings). Apple Silicon (MLX)
only. Defaults to `Qwen3-VL-Embedding-2B` (image embeddings).

It runs a heavy MLX model in a dedicated process (or on a separate machine) so
your application only needs a thin HTTP client.

Repository: <https://github.com/kiarina/mlx-embedding-server>

| Property | Value |
|----------|-------|
| Default model | `mlx-community/Qwen3-VL-Embedding-2B-mxfp8` |
| Output | One embedding vector per image (2048 dims for Qwen3-VL) |
| Default port | 8900 |
| Platform | Apple Silicon (MLX) |

## Requirements

- **macOS on Apple Silicon (M1 or later).** MLX runs on Apple Silicon GPUs;
  Intel Macs, Linux, and Windows are not supported.
- **Python 3.11+** (the repo pins 3.12 via `.python-version`).
- [uv](https://docs.astral.sh/uv/) for dependency management.
- Network access to Hugging Face on first run to download the model.

## Setup & run

Uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python server.py
```

The model is downloaded from Hugging Face on first start. With
`MLX_EMBEDDING_PRELOAD=1` (the default), the model is loaded at startup.

## Endpoints

```text
POST /embed   {"image_base64": "<png/jpeg base64>", "model_id": "<optional>"}
                -> {"embedding": [...], "model_id": "...", "dimension": N}
GET  /health  -> {"status": "ok", "model_id": "..."}
```

### Example

A small `sample.png` is included in this repo so the snippet runs as-is.

```sh
IMAGE_B64=$(base64 -i sample.png)
curl -s http://localhost:8900/embed \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\": \"$IMAGE_B64\"}" | jq '.dimension'
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_EMBEDDING_MODEL_ID` | `mlx-community/Qwen3-VL-Embedding-2B-mxfp8` | Model to load |
| `MLX_EMBEDDING_TEMPLATE_SOURCE` | `Qwen/Qwen3-VL-Embedding-2B` | Source repo for the chat template |
| `MLX_EMBEDDING_HOST` | `0.0.0.0` | Bind host |
| `MLX_EMBEDDING_PORT` | `8900` | Bind port |
| `MLX_EMBEDDING_PRELOAD` | `1` | Load the model at startup |

## Design notes

- MLX arrays/streams are thread-affine, so model loading and inference are all
  funneled through a single `ThreadPoolExecutor(max_workers=1)`. Running FastAPI's
  sync endpoints on an arbitrary threadpool worker breaks stream affinity with
  `RuntimeError: There is no Stream(gpu, N) in current thread`.
- An MLX-converted processor may be missing the `chat_template` and
  `image_ids` / `video_ids` attributes, so `_ensure_model()` loads the chat
  template from the source model and patches in the token ids (a known quirk with
  Qwen3-VL).

## Client configuration

Use any HTTP client and point its `base_url` at this server (default
`http://localhost:8900`, or the host/IP of the machine running it).

## License

MIT — see [LICENSE](LICENSE).
