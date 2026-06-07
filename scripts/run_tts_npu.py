"""piper-plus MS-iSTFT (light dil6) 全NPU 推論: axengine (LLM8850 / NPU3 U16).

全5 chunk (emb_lang / encp / dp / flow / decoder) を **すべて NPU axmodel** で実行する
full-NPU 版。通常の piper-plus は dp / flow が U16 で破綻するため CPU FP32 に残す hybrid
構成になるが、この MS-iSTFT dil6 モデルは multistream 化 + dilation 6 への適応 FT により
**dp / flow / decoder を含む全 chunk を U16 NPU 化済み** (AX620E NPU2 で全PASS、
LLM8850 / AX650 NPU3 ビルドが本セット)。

  uv run python scripts/run_tts_npu.py                          # LLM8850 / AX650 (既定)
  uv run python scripts/run_tts_npu.py --axmodel-dir axmodel/ax620e   # AX620E (NPU2)

provider:
  axclrt   = LLM8850 PCIe ホスト経由
  axengine = SoC native (M4N-Dock / LLM8850 等)
  auto     = pyengine 自動選択 (デフォルト)

ディレクトリ構成 (bundle root = このファイルの親の親):
  axmodel/ax650/   LLM8850 / AX650N NPU3 用 (既定)   config/    config.json + tokens.npz
  axmodel/ax620e/  AX620E NPU2 用                     ref_wav/   参照 wav
  scripts/   この harness + pipeline 一式

参照: ref_wav/baseline_fp32_tokens.wav (tokens.npz と同一発話の fp32 ベースライン)。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import tts_pipeline as P

# 全 chunk を NPU で回す (base hybrid と違い dp/flow も axmodel)
ALL_NPU_CHUNKS = ["emb_lang", "encp", "dp", "flow", "decoder"]

# bundle root (scripts/ の親) を基準に既定パスを解決 → cwd 非依存で実行できる
ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", default=str(ROOT / "config" / "tokens.npz"),
                   help="入力トークン npz (default: config/tokens.npz)")
    p.add_argument("--axmodel-dir", default=str(ROOT / "axmodel" / "ax650"),
                   help="*.axmodel の dir (default: axmodel/ax650/。AX620E は axmodel/ax620e/)")
    p.add_argument("--provider", choices=["axclrt", "axengine", "auto"], default="auto")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--noise-scale",  type=float, default=None,
                   help="未指定なら meta から (config.json inference 値)")
    p.add_argument("--length-scale", type=float, default=None,
                   help="未指定なら meta から")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--out", default=str(ROOT / "out_allnpu.wav"))
    args = p.parse_args()

    tokens, meta = P.load_tokens(args.tokens)
    SR = meta["sampling_rate"]
    print(f"[tokens] text='{meta.get('text','')}' "
          f"PHONE_LEN={meta['phone_len']} MAX_PH={meta['max_ph']} "
          f"MAX_T={meta['max_t']} SR={SR}")

    axe_dir = Path(args.axmodel_dir)
    # 全 chunk を axmodel で load (find_axmodel は *{chunk}[-_]*.axmodel を glob)
    paths = {c: P.find_axmodel(axe_dir, c) for c in ALL_NPU_CHUNKS}
    sessions = P.load_sessions(
        paths, axe_provider=args.provider, axe_device_id=args.device_id,
    )
    print(f"[bench] backend=axengine (LLM8850 / NPU3 U16, FULL-NPU: dp/flow も NPU)  "
          f"provider={args.provider}  runs={args.runs} warmup={args.warmup}")

    noise_scale  = args.noise_scale  if args.noise_scale  is not None else meta.get("noise_scale", 0.667)
    length_scale = args.length_scale if args.length_scale is not None else meta.get("length_scale", 1.0)

    rng = np.random.default_rng(args.seed)
    audio, times, y_len = P.run_pipeline(
        sessions, tokens, meta,
        params=dict(noise_scale=noise_scale, length_scale=length_scale),
        rng=rng, warmup=args.warmup, runs=args.runs,
    )

    wavfile.write(args.out, SR, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    dur = len(audio) / SR
    ws = P.wav_stats(audio, SR)
    print(f"[wav] {args.out}  dur={dur:.2f}s  peak={ws['peak']}  "
          f"RMS={ws['rms']:.4f}  y_len={y_len}/{meta['max_t']}")

    P.print_timing(times, args.runs, args.warmup,
                   backend_label="axengine (LLM8850 / NPU3 U16, FULL-NPU) per-chunk time",
                   dur=dur)
    print()
    print("WAV quality:")
    P.print_quality("allnpu", ws, dur)


if __name__ == "__main__":
    main()
