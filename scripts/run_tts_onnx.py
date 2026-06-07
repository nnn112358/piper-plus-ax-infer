"""piper-plus MS-iSTFT (npu_opt) fp32 推論: onnxruntime CPU.

onnx/ の fp32 ONNX 5本 (emb_lang / encp / dp / flow / decoder) を **全 chunk
onnxruntime CPU** で実行する参照版。run_tts_npu.py (全NPU axengine) の fp32 ベースラインで、
  - NPU 出力との cos / 音質比較の基準
  - NPU カードが無い環境での動作確認
に使う。pipeline ロジック・alignment・math は run_tts_npu.py と完全に共通
(tts_pipeline.Session が拡張子で .onnx→onnxruntime / .axmodel→axengine を自動 dispatch)。

  uv run python scripts/run_tts_onnx.py             # bundle root から (パス自動解決)
  uv run python run_tts_onnx.py                      # scripts/ 内から実行しても可

依存: onnxruntime, numpy, scipy。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import tts_pipeline as P

ALL_CHUNKS = ["emb_lang", "encp", "dp", "flow", "decoder"]

# bundle root (scripts/ の親) 基準で既定パスを解決 → cwd 非依存で実行できる
ROOT = Path(__file__).resolve().parent.parent


def find_onnx(onnx_dir: Path | str, chunk: str) -> Path:
    """onnx/ から chunk に対応する .onnx を引く (末尾 `<chunk>.onnx` で一意)。"""
    d = Path(onnx_dir)
    p = next(d.glob(f"*{chunk}.onnx"), None)
    if p is None:
        raise FileNotFoundError(f"onnx not found for chunk={chunk!r} in {d}")
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", default=str(ROOT / "config" / "tokens.npz"),
                   help="入力トークン npz (default: config/tokens.npz)")
    p.add_argument("--onnx-dir", default=str(ROOT / "onnx"),
                   help="*.onnx の dir (default: onnx/)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--noise-scale",  type=float, default=None,
                   help="未指定なら meta から (config.json inference 値)")
    p.add_argument("--length-scale", type=float, default=None,
                   help="未指定なら meta から")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--out", default=str(ROOT / "out_onnx_fp32.wav"))
    args = p.parse_args()

    tokens, meta = P.load_tokens(args.tokens)
    SR = meta["sampling_rate"]
    print(f"[tokens] text='{meta.get('text','')}' "
          f"PHONE_LEN={meta['phone_len']} MAX_PH={meta['max_ph']} "
          f"MAX_T={meta['max_t']} SR={SR}")

    onnx_dir = Path(args.onnx_dir)
    paths = {c: find_onnx(onnx_dir, c) for c in ALL_CHUNKS}
    sessions = P.load_sessions(paths)   # .onnx → onnxruntime CPU 自動 dispatch
    print(f"[bench] backend=onnxruntime CPU FP32 (全5 chunk)  "
          f"runs={args.runs} warmup={args.warmup}")

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
                   backend_label="onnxruntime CPU FP32 per-chunk time", dur=dur)
    print()
    print("WAV quality:")
    P.print_quality("onnx_fp32", ws, dur)


if __name__ == "__main__":
    main()
