# piper-plus-ax

![license](https://img.shields.io/badge/license-MIT-lightgrey)
![voice](https://img.shields.io/badge/voice-%E3%81%A4%E3%81%8F%E3%82%88%E3%81%BF%E3%81%A1%E3%82%83%E3%82%93-ff8fab)

`piper-plus`(decoder:MC-iSTFT）を NPUコンパイラ`pulsar2` で `*.axmodel` に変換し、NPU 実行**できるようにしたデプロイセットです。
通常の piper-plus からNPUに最適化したモデル構成に変更し、つくよみちゃんコーパスでファインチューニング学習しています。

- 対応: **M5Stack LLM8850** 、**LLM630** 。
- alignment のみ CPU（VITS 共通の非 NN 処理）。

## 構成

```
piper-plus-ax/
├── axmodel/{ax650,ax620e}/   全5 chunk U16 axmodel — これだけで実機推論可
├── config/                   config.json + tokens.npz ("音声合成のテストです")
├── ref_wav/                  out_allnpu.wav (NPU) / out_onnx_fp32.wav (fp32) / stft_npu_vs_fp32.png
└── scripts/                  run_tts_npu.py / run_tts_onnx.py / plot_stft_eval.py / tts_pipeline.py / cpu_alignment.py
```


## クイックスタート

依存は `pyproject.toml` 管理。[`uv`](https://docs.astral.sh/uv/) が初回に `.venv` を自動構築します（bundle root から実行）。

```bash
# 全NPU 推論（実機。pyaxengine が必要）
uv run python scripts/run_tts_npu.py                              # LLM8850（既定）
uv run python scripts/run_tts_npu.py --axmodel-dir axmodel/ax620e  # LLM630

```

> NPU backend は **[pyaxengine](https://github.com/AXERA-TECH/pyaxengine)**（`import axengine`）。PyPI に無く、GitHub Releases の wheel かデバイスイメージ同梱の python を使う。
> fp32 参照 `run_tts_onnx.py` を動かすには別途 `onnx/`（5本）が必要。

## STFT 評価（量子化精度）

同梱 tokens 発話「音声合成のテストです」の **全NPU U16 出力**（`out_allnpu.wav`）と **fp32 参照**（`out_onnx_fp32.wav`）のスペクトログラム比較。

![STFT eval](ref_wav/stft_npu_vs_fp32.png)

NPU U16 出力は fp32 とほぼ同一のスペクトル構造を保ちます（STFT-magnitude cos ≈ **0.93**）。

## モデル / chunk

piper-plus は VITS を 5 チャンクに分割します（`emb_lang` があるのが piper-plus 系の識別点）。

| chunk | 役割 |
|---|---|
| `emb_lang` | 言語/話者埋め込み |
| `enc_p`    | テキストエンコーダ |
| `dp`       | duration predictor |
| `flow`     | normalizing flow |
| `decoder`  | MS-iSTFT vocoder |

## クレジット / ライセンス
- モデル（`axmodel/`）は piper-plus（MIT）由来。ただし**音声はつくよみちゃんコーパスの規約が優先**して適用されます。
- 
- **音声: つくよみちゃんコーパス**（CV: 夢前黎 / つくよみちゃんプロジェクト）— 学習音声データ。<https://tyc.rei-yumesaki.net/material/corpus/>
  > ⚠️ 合成音声・モデルの利用は **つくよみちゃんコーパスの利用規約**に従ってください（クレジット表記例:「つくよみちゃんコーパス（CV.夢前黎）」、禁止用途あり）。公開・配布・利用の前に必ず[公式の最新規約](https://tyc.rei-yumesaki.net/material/corpus/)を確認すること。
- **piper-plus**（ayousanz）— VITS + iSTFT TTS（MIT）。<https://github.com/ayousanz/piper-plus>
- **AXera pulsar2** — NPU コンパイラ。
- **[pyaxengine](https://github.com/AXERA-TECH/pyaxengine)**（AXERA-TECH）— axmodel の Python 推論ランタイム。

