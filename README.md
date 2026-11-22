# piper-plus-ax

![license](https://img.shields.io/badge/license-MIT-lightgrey)
![voice](https://img.shields.io/badge/voice-%E3%81%A4%E3%81%8F%E3%82%88%E3%81%BF%E3%81%A1%E3%82%83%E3%82%93-ff8fab)

[piper-plus](https://github.com/ayutaz/piper-plus) を AXera NPU 搭載ボード **M5Stack LLM8850**（AX650 NPU3）と **LLM630**（AX620E NPU2）で動かすためのデプロイセットです。
モデル構成を NPU 向けに変え、つくよみちゃんコーパスでファインチューニングしています。

- VITS を 5 つの chunk に分け、全 chunk を NPU（U16 量子化）で実行します。
- CPU で動くのは alignment（VITS 共通の NN 以外の処理）だけです。
- fp32 ONNX との比較で、cos ≈ 0.979（ep599, AX620E / AX650）の精度です。

## 目次

- [ディレクトリ構成](#ディレクトリ構成)
- [クイックスタート](#クイックスタート)
- [評価](#評価)
- [モデル](#モデル)
- [Decoder の変更](#decoder-の変更)
- [クレジット / ライセンス](#クレジット--ライセンス)

## ディレクトリ構成

```
piper-plus-ax/
├── axmodel/
│   ├── ax650/     LLM8850 / AX650 NPU3 用 U16 axmodel（5 chunk）— 既定
│   └── ax620e/    LLM630 / AX620E NPU2 用 U16 axmodel（5 chunk）
├── onnx/          fp32 参照用 ONNX（5 chunk）
├── config/        config.json + tokens.npz（発話「音声合成のテストです」）
├── scripts/
│   ├── run_tts_npu.py     全 NPU 推論（実機）
│   ├── run_tts_onnx.py    fp32 ONNX 参照推論（ホスト）
│   ├── plot_stft_eval.py  NPU 出力と fp32 参照の STFT 比較図
│   ├── tts_pipeline.py    5 chunk の推論パイプライン
│   └── cpu_alignment.py   alignment の CPU 実装
├── ref_wav/       out_allnpu.wav / out_onnx_fp32.wav / stft_npu_vs_fp32.png
└── demo/          ep599 の合成音声と STFT 比較図
```

chunk は `emb_lang` / `encp` / `dp` / `flow` / `decoder` の 5 つです。

## クイックスタート

依存は `pyproject.toml` で管理しています。リポジトリのルートで [`uv`](https://docs.astral.sh/uv/) を実行すると、初回に `.venv` が作られます。

### NPU 推論（実機）

```bash
uv run python scripts/run_tts_npu.py                               # LLM8850（既定: axmodel/ax650）
uv run python scripts/run_tts_npu.py --axmodel-dir axmodel/ax620e  # LLM630
```

NPU の実行には [pyaxengine](https://github.com/AXERA-TECH/pyaxengine)（`import axengine`）が必要です。
PyPI では配布されていないため、GitHub Releases の wheel を入れるか、デバイスイメージに入っている Python を使ってください。

```bash
uv run --python <device-python> python scripts/run_tts_npu.py
```

主なオプション:

| オプション | 既定値 | 内容 |
|---|---|---|
| `--axmodel-dir` | `axmodel/ax650` | 使う axmodel のディレクトリ |
| `--provider` | `auto` | `axclrt`（LLM8850 PCIe ホスト経由）/ `axengine`（SoC native）/ `auto` |
| `--tokens` | `config/tokens.npz` | 入力トークン |
| `--noise-scale` / `--length-scale` | config の値 | 合成パラメータ |
| `--seed` | `20260528` | noise の乱数シード |
| `--out` | `out_allnpu.wav` | 出力 wav |

### fp32 参照推論と評価図（ホスト）

```bash
uv run python scripts/run_tts_onnx.py    # onnx/ を使って fp32 で合成
uv run python scripts/plot_stft_eval.py  # ref_wav/ の 2 つの wav から STFT 比較図を作る
```

## 評価

### STFT 比較（同梱 tokens）

同梱 tokens の発話「音声合成のテストです」について、全 NPU U16 出力（`ref_wav/out_allnpu.wav`）と fp32 参照（`ref_wav/out_onnx_fp32.wav`）のスペクトログラムを比べました。
NPU U16 出力のスペクトル構造は fp32 とほぼ同じです（STFT 振幅の cos ≈ **0.93**）。

<p align="center">
  <img width="600" alt="STFT eval" src="ref_wav/stft_npu_vs_fp32.png" />
</p>

### 最新検証（ep599, 2026-06-09）

「こんにちは。つくよみちゃんです。」を AX620E NPU2 / AX650 NPU3 の axmodel で全 NPU 合成し（sim_axengine = pulsar2 cmodel backend）、fp32 ONNX と E2E で比べました。

| ターゲット | cos (vs FP32) | 8268.75 Hz Δ | 無音 / beep |
|---|---|---|---|
| AX620E NPU2 U16 | **0.97922** | +1.6 dB | なし |
| AX650 NPU3 U16  | **0.97909** | +1.6 dB | なし |

- decoder 単体（実際の z 入力、cmodel vs ONNX FP32）: AX620E で cos **0.9998**
- 合成音声: `demo/tyc_ep599_ax620e.wav` / `demo/tyc_ep599_ax650.wav`

<p align="center">
  <img width="600" alt="ep599 STFT 3-way" src="demo/ep599_stft_3way.png" />
</p>

## モデル

NPU コンパイラ（Pulsar2）で扱えるよう、モデル構造を変えています。

| 項目 | piper-plus (fp16) | piper-plus-ax (npu_opt) |
|------|-------------------|-------------------------|
| 構成 | ONNX 1 ファイル | ONNX 5 分割（emb_lang / encp / dp / flow / decoder） |
| 精度・形状 | FP16・動的 | FP32・固定（PH=256 / T=512） |
| 動的 op・noise | グラフ内 | 動的 op は CPU へ、noise は外部入力 |
| NPU 非対応 op | グラフ内 | 置換 |
| decoder | MB-iSTFT（polar head: Exp/Sin/Cos） | MS-iSTFT + **Cartesian head** |
| resblock dilation | 最大 12 | **最大 6** |
| 実行先 | CPU / GPU | LLM8850 / LLM630 |

### NPU 非対応 op の置換

| NPU 非対応 op | 置換先 |
|---|---|
| `NonZero` / `GatherND` / `ScatterND` | `torch.where` |
| `ScatterND` | slice + concat |
| `GatherElements` | onehot × Mul × ReduceSum |
| `torch.cumsum` | Concat + Add |
| `RandomNormalLike` | 外部入力（z_p） |
| `Range` / `NonZero` / `ScatterND`（alignment） | CPU 実装（`scripts/cpu_alignment.py`） |
| `Erf` | GELU |

## Decoder の変更

### MB-iSTFT の beep と MS-iSTFT

<p align="center">
  <img width="640" alt="MB-iSTFT と MS-iSTFT の構成比較" src="https://github.com/user-attachments/assets/212c908c-212b-4813-89f7-10cdbbd32216" />
</p>

元の MB-iSTFT は、音声を 4 つの周波数サブバンドに分けて生成し、最後に PQMF（学習しない固定係数の合成フィルタ）で 1 本にまとめます。
Pulsar2 で U16 量子化すると、次の流れで beep が出ます。

1. 各サブバンド信号に量子化誤差が乗る。
2. 固定の PQMF を通ると、誤差がサブバンド境界の周波数に集まる（22.05 kHz を 4 分割すると 2756.25 / 5512.5 / 8268.75 Hz）。
3. その周波数に定常的な寄生トーン（ピー音 / whine）が出る。固定フィルタなので、誤差を吸収・分散できない。

<p align="center">
  <img width="500" alt="量子化で発生するピー音の説明図" src="https://github.com/user-attachments/assets/a9f6216d-c266-458c-a018-11e05729231f" />
</p>

MS-iSTFT では、PQMF を学習可能な conv（`multistream_conv_post`）に置き換え、量子化を考慮して再学習します。
合成フィルタが学習で調整されるので、量子化誤差が境界に集まらずに分散します。
これで beep が消え、全 chunk を NPU で実行できるようになりました（decoder 単体で cos ≈ 0.99999）。

### AX620E 向けの対応

AX620E NPU2 では、MS-iSTFT だけでは別の不具合が残るため、次の変更も入れています。

| 変更 | 解決した不具合 | 補足 |
|---|---|---|
| **Cartesian head**（real/imag を直接予測） | **無音**（cos −0.90）。NPU2 の LUT が `exp` / `sin` / `cos` を正しく計算できない | Exp/Sin/Cos が 0 個になる。サイズ・速度・MACs は変わらない |
| **MS-iSTFT** | 8 kHz 付近の **beep**（PQMF と twiddle の U16 量子化） | 前節のとおり |
| **resblock dilation ≤ 8（採用は 6）** | AX620E の `conv_layer_check` で build / run が FAIL | dilation はゼロ詰めなので、MACs・速度・サイズは変わらない |

学習側（`piper-plus_FT`）では、これらを `cartesian` フラグ、`ms_istft_vits`、`resblock_dilation_sizes` の末尾の値で指定します。既存の polar / MB / dil12 の経路も残っています。

## クレジット / ライセンス

### Piper Plus つくよみちゃん

本リポジトリは「[Piper Plus つくよみちゃん](https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan)」（開発: ようさん / [@ayousanz](https://x.com/ayousanz)）を使用しています。

- 「Piper Plus つくよみちゃん」の開発には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データセット「[つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/)（CV.夢前黎）」が使われています。
- つくよみちゃん側から見ると非公式の作品です。夢前黎様は開発・サポートに携わっていません。本リポジトリについてのお問い合わせは、夢前黎様やようさんではなく、本リポジトリの [Issues](https://github.com/nnn112358/piper-plus-ax-infer/issues) へお願いします。

**クレジット表記（必須）**: 生成した音声を動画・配信・講演などで公開するときは、説明欄など視聴者が確認できる場所に次のとおり記載してください。

> ■Piper Plus:つくよみちゃん

**禁止事項**: 出力した音声を次の目的で使用することを禁止します。

- 人を批判・攻撃すること（定義は [つくよみちゃんキャラクターライセンス](https://tyc.rei-yumesaki.net/about/terms/#condition3) に準じます）。
- 特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。
- 刺激の強い表現をゾーニングなしで公開すること。
- 他者に対して二次利用を許可する形で公開すること（出力音声を素材として配布・販売する、出力音声で新たな音声合成ソフト類を作って他者に使わせるなど、つくよみちゃん（CV.夢前黎）の声質を他者が利用できるようにする行為全般を含みます）。

鑑賞用の作品として配布・販売することは問題ありません。商用利用は個人・法人を問わず制限はありません。
改変・再配布する場合は、つくよみちゃんコーパスの利用規約「[③声質を使用した音声合成ソフト等を公開する](https://tyc.rei-yumesaki.net/material/corpus/#terms3)」に従ってください。詳細は必ず [Piper Plus つくよみちゃんのモデルカード](https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan) と[公式規約](https://tyc.rei-yumesaki.net/material/corpus/)の最新版を確認してください。

### ライセンス

- `scripts/` のコードは MIT License です（[LICENSE](LICENSE)）。
- モデル（`axmodel/`、`onnx/`）は piper-plus（MIT）由来です。ただし音声の利用には、上記のつくよみちゃんコーパスの規約が優先して適用されます。

### 使用しているソフトウェア・データ

- **つくよみちゃんコーパス**（CV: 夢前黎 / つくよみちゃんプロジェクト）— 学習音声データ。<https://tyc.rei-yumesaki.net/material/corpus/>
- **[piper-plus](https://github.com/ayutaz/piper-plus)**（ayousanz）— VITS + iSTFT TTS（MIT）。
- **AXera Pulsar2** — NPU コンパイラ。
- **[pyaxengine](https://github.com/AXERA-TECH/pyaxengine)**（AXERA-TECH）— axmodel の Python 推論ランタイム。
