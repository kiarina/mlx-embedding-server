# mlx-embedding-server

*[English](README.md) | 日本語*

`mlx-embeddings` で動くマルチモーダル埋め込みを、HTTP 経由で使うための単機能サーバ。
Apple Silicon (MLX) 専用。既定モデルは `Qwen3-VL-Embedding-2B`（画像埋め込み）。

重い MLX モデルを別プロセス（別マシン）で常駐させ、アプリ側は薄い HTTP クライアントから
叩く、という構成を想定しています。

| Property | Value |
|----------|-------|
| Default model | `mlx-community/Qwen3-VL-Embedding-2B-mxfp8` |
| Output | 画像 1 枚あたりの埋め込みベクトル（Qwen3-VL は 2048 次元） |
| Default port | 8900 |
| Platform | Apple Silicon (MLX) |

## セットアップ & 起動

[uv](https://docs.astral.sh/uv/) を使います。

```sh
uv sync
uv run python server.py
```

初回起動時に Hugging Face からモデルがダウンロードされます。
`MLX_EMBEDDING_PRELOAD=1`（既定）で起動時にモデルをロードします。

## エンドポイント

```text
POST /embed   {"image_base64": "<png/jpeg base64>", "model_id": "<optional>"}
                -> {"embedding": [...], "model_id": "...", "dimension": N}
GET  /health  -> {"status": "ok", "model_id": "..."}
```

### 例

スニペットがそのまま動くよう、小さな `sample.png` をリポジトリに同梱しています。

```sh
IMAGE_B64=$(base64 -i sample.png)
curl -s http://localhost:8900/embed \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\": \"$IMAGE_B64\"}" | jq '.dimension'
```

## 環境変数

| Variable | Default | 説明 |
|----------|---------|------|
| `MLX_EMBEDDING_MODEL_ID` | `mlx-community/Qwen3-VL-Embedding-2B-mxfp8` | ロードするモデル |
| `MLX_EMBEDDING_TEMPLATE_SOURCE` | `Qwen/Qwen3-VL-Embedding-2B` | chat template の取得元 |
| `MLX_EMBEDDING_HOST` | `0.0.0.0` | バインドするホスト |
| `MLX_EMBEDDING_PORT` | `8900` | バインドするポート |
| `MLX_EMBEDDING_PRELOAD` | `1` | 起動時にモデルをロードするか |

## 設計メモ

- MLX の配列/ストリームはスレッドアフィニティを持つため、モデルのロードと推論を
  すべて単一の `ThreadPoolExecutor(max_workers=1)` に集約しています。FastAPI の sync
  endpoint を任意のスレッドプールで動かすと
  `RuntimeError: There is no Stream(gpu, N) in current thread` になります。
- MLX 変換済み processor には `chat_template` や `image_ids` / `video_ids` が欠けている
  ことがあるため、`_ensure_model()` で元モデルの chat template を読み込み、token id
  属性を補っています（Qwen3-VL での既知の挙動）。

## クライアント側の設定例

任意の HTTP クライアントから利用できます。`base_url` をこのサーバに向けてください
（既定 `http://localhost:8900`、別マシンならその IP/ホスト名）。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。
