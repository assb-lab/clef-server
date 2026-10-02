# clef-server

[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) / [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
の推論サーバーと Python クライアントのモノレポです。
サーバーは Jev / SystemOne 互換の `POST /v1/systemone` を FastAPI で公開します。

Clef は「状態 (state)」と「型付きの質問 (questions)」を受け取り、各質問の選択肢ごとの確率を 1 回の forward で返す判定モデルです（テキストは生成しません）。

```
.
├── server/                  # clef-server: 推論サーバー (torch / transformers / FastAPI)
│   └── src/clef_server/
├── client/                  # clef-client: Python クライアント (requests のみに依存)
│   ├── src/clef_client/
│   └── examples/
│       ├── basic.py         # 問い合わせの振り分け (choice / score / noul)
│       ├── image.py         # 画像付きの判定
│       ├── cif_structure.py # CIF が指定した構造型かどうかを判定
│       └── data/*.cif       # サンプル CIF (NaCl, CsCl, SrTiO3)
└── pyproject.toml           # uv workspace
```

## モデル

| `--model` | 中身 | 重みのサイズ (bf16) | 目安のメモリ |
|---|---|---|---|
| `clef` (デフォルト) | Qwen3.8-27B ベース | 約 55GB | VRAM 80GB 以上 (H100 / H200) |
| `clef-flash` | Qwen3.5-9B ベース | 約 19GB | VRAM 24GB 以上 / Apple Silicon 32GB 以上 |

Decision Index ではタスクによってどちらが良いかが違います。clef-flash のほうがレイテンシは約 5 倍短いです（中央値 39ms 対 209ms）。

## セットアップ

[uv](https://docs.astral.sh/uv/) を使います。Python は 3.14 です。

```bash
# サーバーとクライアントを両方入れる (GPU マシン)
uv sync

# クライアントだけ入れる (torch 不要)
uv sync --package clef-client
```

モデルは初回起動時に Hugging Face から自動でダウンロードされ、`~/.cache/huggingface` にキャッシュされます。事前に取得しておく場合:

```bash
uv run hf download Cloudflare/clef-flash
```

## サーバーの起動

```bash
uv run clef-server                       # Clef (27B)
uv run clef-server --model clef-flash    # Clef-Flash (9B)
```

起動してモデルのロードが終わると `http://0.0.0.0:8000` で待ち受けます。

### オプション

| 引数 | 環境変数 | デフォルト | 説明 |
|---|---|---|---|
| `--model`, `-m` | `CLEF_MODEL` | `clef` | `clef` / `clef-flash` / 任意の HF リポジトリ ID / ローカルパス |
| `--revision` | `CLEF_REVISION` | 最新 | モデルのリビジョン (commit SHA など) |
| `--device` | `CLEF_DEVICE` | 自動 (`cuda` → `mps` → `cpu`) | `cuda:1` なども指定可 |
| `--dtype` | `CLEF_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--max-length` | `CLEF_MAX_LENGTH` | `16384` | 入力の最大トークン数 |
| `--host` | `HOST` | `0.0.0.0` | 待ち受けるアドレス |
| `--port`, `-p` | `PORT` | `8000` | 待ち受けるポート |

引数と環境変数の両方を指定した場合は引数が優先されます。`HF_TOKEN` / `HF_HOME` もそのまま使えます。

```bash
uv run clef-server -m clef-flash --device cuda:1 -p 9000
```

GPU 1 枚にモデル 1 つの構成なので、推論は 1 リクエストずつ順に処理します。並列に捌きたい場合は GPU ごとにプロセスを立てて (`--device cuda:N -p 900N`)、前段にロードバランサを置いてください。

## クライアント

```python
from clef_client import ClefClient, choice, noul, score

client = ClefClient("http://localhost:8000")
print(client.health())  # {'status': 'ok', 'model': 'Cloudflare/clef-flash', 'device': 'cuda'}

response = client.systemone(
    "Our checkout started returning errors and orders are blocked.",
    {
        "department": choice(
            {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            "Which team should handle the message?",
        ),
        "urgency": score(["Can wait", "This week", "Today"]),
        "outage": noul("Is a service down?"),
    },
)
print(response["answers"]["department"]["choice"])  # technical
```

- `client.systemone(state, questions, images=..., videos=...)` はレスポンス全体 (`model`, `answers`, `usage`) を返します。
- `client.answers(...)` は `answers` の部分だけを返します。
- `images` にはローカルのパス、`bytes`、http(s) の URL を混ぜて渡せます。ローカルのものは base64 にして送ります。
- サーバーがエラーを返した場合や接続できなかった場合は `ClefError` が送出されます。HTTP ステータスは `.status_code` で取れます。

`noul` / `choice` / `score` は質問の dict を作るためのヘルパーです。dict を直接書いても構いません。

### requests で直接叩く場合

```python
import requests

response = requests.post("http://localhost:8000/v1/systemone", json={
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
})
response.raise_for_status()
print(response.json()["answers"])
```

### curl で叩く場合

```bash
curl -s localhost:8000/health

curl -s localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
      },
      "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
      "outage": {"type": "noul", "instructions": "Is a service down?"}
    }
  }'
```

画像付きのリクエスト (base64 で埋め込む):

```bash
IMG=$(base64 < receipt.jpg | tr -d '\n')
curl -s localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d "{
    \"state\": {\"task\": \"Review the attached receipt.\"},
    \"images\": [\"$IMG\"],
    \"questions\": {\"legible\": {\"type\": \"noul\", \"instructions\": \"Is the receipt total legible?\"}}
  }"
```

画像が大きいとコマンドラインの長さ制限に引っかかります。その場合は JSON をファイルに書き出して `-d @request.json` で渡してください。

### サンプル

```bash
uv run client/examples/basic.py
uv run client/examples/basic.py "Please refund my last invoice." --url http://gpu-host:8000
uv run client/examples/image.py receipt.jpg --question "Is the receipt total legible?"
```

## 例: CIF の構造型を判定する (`cif_structure.py`)

CIF ファイルと構造型の名前を引数で渡すと、その結晶構造が指定した構造型に属するかを判定します。

```bash
uv run client/examples/cif_structure.py client/examples/data/NaCl.cif --structure rocksalt
uv run client/examples/cif_structure.py client/examples/data/*.cif -s perovskite
uv run client/examples/cif_structure.py my.cif -s "Ruddlesden-Popper" --hint "A(n+1)B(n)O(3n+1), perovskite slabs separated by rock-salt layers"
```

```
client/examples/data/NaCl.cif
  ClNa  F m -3 m (#225)  Z=4
    Na1    CN6  Clx6           2.820-2.820 Å
    Cl1    CN6  Nax6           2.820-2.820 Å
  rocksalt? YES  (confidence 0.97, relation=prototype, match 3.81/4)
  checks: stoichiometry=0.98  symmetry=0.97  coordination=0.96
  (231 ms)
```

上の出力の数値は例です。

### 処理の流れ

1. **クライアント側で CIF を解析する**（依存ライブラリなし）
   - 対称操作を非対称単位に適用して単位胞内の全原子を求め、組成・組成式・Z を出します。
   - 各サイトについて、第一配位圏（最短距離の 1.15 倍以内）の配位数・隣接元素・距離を計算します。
2. **抽出した特徴量と生の CIF を `state` に入れて、異なる観点の質問を 1 リクエストでまとめて送る**

   | 質問 ID | 型 | 内容 |
   |---|---|---|
   | `is_target` | noul | 指定した構造型に属するか（元素の置換は許容） |
   | `relation` | choice | prototype / distorted / ordered_superstructure / related_family / unrelated |
   | `match_level` | score | 一致度 (0–4) |
   | `stoichiometry_ok` | noul | 組成比が構造型と整合するか |
   | `symmetry_ok` | noul | 空間群が構造型のもの、またはその部分群か |
   | `coordination_ok` | noul | 配位数と隣接元素が構造型と整合するか |
   | `crystal_system` | choice | 晶系（モデルが入力を正しく読めているかの確認用） |
   | `data_quality` | choice | complete / disordered / insufficient |

3. **クライアント側で回答を突き合わせて判定する**
   - `is_target` が `--threshold` 以上なら YES とします。
   - 次のような食い違いは `!` 付きで警告します。
     - YES なのに下位のチェックが落ちている、または関係が unrelated になっている
     - 晶系の回答が CIF の空間群番号と合わない
     - 部分占有があるなど、入力に問題がある

### 引数

| 引数 | 説明 |
|---|---|
| `cif` (複数可) | CIF ファイル |
| `--structure`, `-s` | 構造型の名前 (必須) |
| `--hint` | 構造型の定義文。rocksalt, cscl, zincblende, wurtzite, fluorite, antifluorite, rutile, perovskite, spinel, corundum, nias, diamond, fcc, bcc, hcp は組み込みの定義を使います |
| `--threshold` | YES と判定する確率の閾値 (デフォルト 0.5) |
| `--no-raw` | 生の CIF は送らず、解析した特徴量だけを送る |
| `--max-raw-chars` | これより大きい CIF は生のテキストを送らない (デフォルト 12000) |
| `--json` | レポートの代わりに JSON を出力する |
| `--url` | サーバーの URL (デフォルト `http://localhost:8000`) |

## API リファレンス

### `POST /v1/systemone`

| フィールド | 必須 | 説明 |
|---|---|---|
| `state` | ✓ | 判定対象の状況。文字列でも任意の JSON でも可 |
| `questions` | ✓ | 質問 ID → 質問 のマップ |
| `model` | | レスポンスにそのまま返る名前。省略時はモデル名 |
| `images` | | 画像のリスト。各要素は http(s) URL / data URI / base64 文字列 |
| `videos` | | 動画のリスト。各動画はフレーム画像 (形式は `images` と同じ) のリスト |
| `media_kwargs` | | 画像・動画プロセッサに渡す追加の引数 |

| 質問の `type` | `criteria` | 回答 |
|---|---|---|
| `noul` | 省略可。`{"true": "...", "false": "..."}` | `noul`: true である確率 |
| `choice` | 選択肢 ID → 説明 のマップ | `choice`, `confidence`, `probabilities` |
| `score` | 順序付きの説明のリスト (0 始まり) | 期待値 `score`, `confidence`, `legend`, `probabilities` |

`instructions` は省略でき、省略すると質問 ID が指示として使われます。`usage` には `input_tokens` と、サーバー側の推論時間 `latency_ms` が入ります。

レスポンスの例 (数値は例):

```json
{
  "model": "clef",
  "answers": {
    "department": {
      "type": "choice", "choice": "technical", "confidence": 0.98,
      "probabilities": {"billing": 0.02, "technical": 0.98}
    },
    "urgency": {
      "type": "score", "score": 1.93, "confidence": 0.94,
      "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
      "probabilities": {"0": 0.01, "1": 0.05, "2": 0.94}
    },
    "outage": {"type": "noul", "noul": 0.91}
  },
  "usage": {"input_tokens": 142, "output_tokens": 0, "latency_ms": 210.4}
}
```

ステータスコード:
- `400`: 質問が空、`type` が不正、`criteria` が空、画像をデコードできない
- `422`: ボディが JSON オブジェクトではない

### `GET /health`

`{"status": "ok" | "loading", "model": ..., "device": ...}` を返します。`/docs` では Swagger UI が開きます。

## ライセンス

このリポジトリは MIT です。Clef / Clef-Flash の重みは Apache-2.0 です。
