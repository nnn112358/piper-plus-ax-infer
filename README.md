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

## モデル
piper-plus-ax は NPUコンパイラ`に最適化したモデル構造に変更しています。
VITS を 5つのonnxに分割します（`emb_lang` があるのが piper-plus 系の識別点）。

| 項目 | piper-plus (fp16) | piper-plus-ax (npu_opt) |
|------|---------------|---------------------|
| 構成 | onnx1ファイル | onnx5分割（emb_lang/encp/dp/flow/decoder） |
| 精度・形状 | FP16・動的 | FP32・固定（PH=256/T=512） |
| 動的op・noise | グラフ内に内包 | 除去→CPU、noiseは外部入力化 |
| NPU非対応op | グラフ内に内包  | 置換 |
| decoder | MB-iSTFT | MS-iSTFT（Beep音対策） |
| 実行先 | CPU/GPU  | LLM8850/LLM630 |

### NPU非対応op
| NPU非対応op | 置換先  |
|--------|------|
| NonZero / GatherND / ScatterND | torch.where |
| ScatterND | slice+concat |
| GatherElements |  onehot×Mul×ReduceSum |
| torch.cumsum  | Concat+Add累積 |
| RandomNormalLike |  外部入力(z_p)化 |
| Range / NonZero / ScatterND | CPU実装(align_cpu.py) |
| Erf | GELU |

###  Decoder:MB-iSTFT /MS-iSTFT
<img width="1446" height="704" alt="image" src="https://github.com/user-attachments/assets/212c908c-212b-4813-89f7-10cdbbd32216" />
  MB-iSTFT（元） は、音声を4つの周波数サブバンドに分けて生成し、最後に
  PQMF（固定の合成フィルタ） で1本に合成します。このフィルタは学習されない固定係
  数で、サブバンド境界（44.1kHz/4分割だと 5512.5 / 8268.75 / 11025
  Hz）に急峻な遷移を持ちます。

  NPUでU16量子化すると：
  - 各サブバンド信号に量子化誤差が乗る
  - それが固定PQMFを通ると、サブバンド境界の特定周波数に誤差が集中・整列
  - → その周波数に**定常的な寄生トーン（ピー音 / whine）**が発生
  - 固定フィルタなので、誤差を吸収・分散できない

  解決策＝MS-iSTFT
  PQMF（固定合成）を、学習可能なconv（multistream_conv_post）に差し替え、量子化
  を意識して再学習。
  合成フィルタが学習で調整可能になるため、量子化誤差を境界に集中させず分散できる
  - → ピー音が消滅、全NPU化が可能（cos ≈ 0.99999）

## クレジット / ライセンス
- モデル（`axmodel/`）は piper-plus（MIT）由来。ただし**音声はつくよみちゃんコーパスの規約が優先**して適用されます。
- 
- **音声: つくよみちゃんコーパス**（CV: 夢前黎 / つくよみちゃんプロジェクト）— 学習音声データ。<https://tyc.rei-yumesaki.net/material/corpus/>
  > ⚠️ 合成音声・モデルの利用は **つくよみちゃんコーパスの利用規約**に従ってください（クレジット表記例:「つくよみちゃんコーパス（CV.夢前黎）」、禁止用途あり）。公開・配布・利用の前に必ず[公式の最新規約](https://tyc.rei-yumesaki.net/material/corpus/)を確認すること。
- **piper-plus**（ayousanz）— VITS + iSTFT TTS（MIT）。<https://github.com/ayousanz/piper-plus>
- **AXera pulsar2** — NPU コンパイラ。
- **[pyaxengine](https://github.com/AXERA-TECH/pyaxengine)**（AXERA-TECH）— axmodel の Python 推論ランタイム。

