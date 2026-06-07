# piper-plus-ax

![license](https://img.shields.io/badge/license-MIT-lightgrey)
![voice](https://img.shields.io/badge/voice-%E3%81%A4%E3%81%8F%E3%82%88%E3%81%BF%E3%81%A1%E3%82%83%E3%82%93-ff8fab)

piper-plus を NPU ボード **LLM8850 / LLM630** で実行できるようにしたデプロイセットです。
通常の piper-plus から NPU に最適化したモデル構成へ変更し、つくよみちゃんコーパスでファインチューニング学習しています。

- 対応: **M5Stack LLM8850**、**LLM630**
- alignment のみ CPU（VITS 共通の非 NN 処理）

## 構成

```
piper-plus-ax/
├── axmodel/{ax650,ax620e}/   全 5 chunk U16 axmodel — これだけで実機推論可
├── config/                   config.json + tokens.npz（"音声合成のテストです"）
├── ref_wav/                  out_allnpu.wav (NPU) / out_onnx_fp32.wav (fp32) / stft_npu_vs_fp32.png
└── scripts/                  run_tts_npu.py / run_tts_onnx.py / plot_stft_eval.py / tts_pipeline.py / cpu_alignment.py
```

## クイックスタート

依存は `pyproject.toml` で管理しています。[`uv`](https://docs.astral.sh/uv/) が初回に `.venv` を自動構築します（bundle root から実行）。

```bash
# 全 NPU 推論（実機。pyaxengine が必要）
uv run python scripts/run_tts_npu.py                               # LLM8850（既定）
uv run python scripts/run_tts_npu.py --axmodel-dir axmodel/ax620e  # LLM630
```

> NPU backend は **[pyaxengine](https://github.com/AXERA-TECH/pyaxengine)**（`import axengine`）。PyPI には無く、GitHub Releases の wheel か、デバイスイメージ同梱の python を使います。
> fp32 参照の `run_tts_onnx.py` を動かすには、別途 `onnx/`（5 本）が必要です。

## STFT 評価（量子化精度）

同梱 tokens の発話「音声合成のテストです」について、**全 NPU U16 出力**（`out_allnpu.wav`）と **fp32 参照**（`out_onnx_fp32.wav`）のスペクトログラムを比較します。

<p align="center">
  <img width="600" alt="STFT eval" src="ref_wav/stft_npu_vs_fp32.png" />
</p>

NPU U16 出力は fp32 とほぼ同一のスペクトル構造を保ちます（STFT-magnitude cos ≈ **0.93**）。

## モデル

piper-plus-ax は NPU コンパイラに最適化したモデル構造へ変更しています。
VITS を 5 つの ONNX に分割します（`emb_lang` があるのが piper-plus 系の識別点）。

| 項目 | piper-plus (fp16) | piper-plus-ax (npu_opt) |
|------|-------------------|-------------------------|
| 構成 | onnx 1 ファイル | onnx 5 分割（emb_lang / encp / dp / flow / decoder） |
| 精度・形状 | FP16・動的 | FP32・固定（PH=256 / T=512） |
| 動的 op・noise | グラフ内に内包 | 除去 → CPU、noise は外部入力化 |
| NPU 非対応 op | グラフ内に内包 | 置換 |
| decoder | MB-iSTFT | MS-iSTFT（Beep 音対策） |
| 実行先 | CPU / GPU | LLM8850 / LLM630 |

### NPU 非対応 op

| NPU 非対応 op | 置換先 |
|---|---|
| `NonZero` / `GatherND` / `ScatterND` | `torch.where` |
| `ScatterND` | slice + concat |
| `GatherElements` | onehot × Mul × ReduceSum |
| `torch.cumsum` | Concat + Add  |
| `RandomNormalLike` | 外部入力（z_p）化 |
| `Range` / `NonZero` / `ScatterND` | CPU 実装（`align_cpu.py`） |
| `Erf` | GELU |

## Decoder: MB-iSTFT / MS-iSTFT

<p align="center">
  <img width="640" alt="MB-iSTFT と MS-iSTFT の構成比較" src="https://github.com/user-attachments/assets/212c908c-212b-4813-89f7-10cdbbd32216" />
</p>

MB-iSTFT（元）は、音声を 4 つの周波数サブバンドに分けて生成し、最後に PQMF（固定の合成フィルタ）で 1 本に合成します。このフィルタは学習されない固定係数で、サブバンド境界（44.1 kHz / 4 分割だと 5512.5 / 8268.75 / 11025 Hz）に急峻な遷移を持ちます。

<p align="center">
  <img width="500" alt="量子化で発生するピー音の説明図" src="https://github.com/user-attachments/assets/a9f6216d-c266-458c-a018-11e05729231f" />
</p>

NPU で U16 量子化すると:

- 各サブバンド信号に量子化誤差が乗る
- それが固定 PQMF を通ると、サブバンド境界の特定周波数に誤差が集中・整列する
- → その周波数に **定常的な寄生トーン（ピー音 / whine）** が発生する
- 固定フィルタなので、誤差を吸収・分散できない

**解決策 = MS-iSTFT**

PQMF（固定合成）を、学習可能な conv（`multistream_conv_post`）に差し替え、量子化を意識して再学習します。合成フィルタが学習で調整可能になるため、量子化誤差を境界に集中させず分散できます。

- → ピー音が消滅し、全 NPU 化が可能（cos ≈ 0.99999）

## クレジット / ライセンス

- モデル（`axmodel/`）は piper-plus（MIT）由来。ただし **音声はつくよみちゃんコーパスの規約が優先** して適用されます。
- **音声: つくよみちゃんコーパス**（CV: 夢前黎 / つくよみちゃんプロジェクト）— 学習音声データ。<https://tyc.rei-yumesaki.net/material/corpus/>
  > ⚠️ 合成音声・モデルの利用は **つくよみちゃんコーパスの利用規約** に従ってください（クレジット表記例:「つくよみちゃんコーパス（CV. 夢前黎）」、禁止用途あり）。公開・配布・利用の前に、必ず [公式の最新規約](https://tyc.rei-yumesaki.net/material/corpus/) を確認してください。
- **piper-plus**（ayousanz）— VITS + iSTFT TTS（MIT）。<https://github.com/ayousanz/piper-plus>
- **AXera pulsar2** — NPU コンパイラ。
- **[pyaxengine](https://github.com/AXERA-TECH/pyaxengine)**（AXERA-TECH）— axmodel の Python 推論ランタイム。
