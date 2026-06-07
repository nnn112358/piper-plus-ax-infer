"""STFT 評価図を生成する: 全NPU 出力 vs fp32 参照 (同一発話).

ref_wav/out_allnpu.wav    (run_tts_npu.py の全NPU U16 出力) と
ref_wav/out_onnx_fp32.wav (run_tts_onnx.py の fp32 参照出力) の
対数振幅スペクトログラムを並べ、U16 量子化による劣化を可視化する。
波形 cos / STFT(振幅) cos も併記し ref_wav/stft_npu_vs_fp32.png に保存。

2 wav は同梱 tokens.npz と同一発話 ("音声合成のテストです")・同一長なので 1:1 比較できる。

  uv run python scripts/plot_stft_eval.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "ref_wav"
N_FFT, HOP = 1024, 256


def load(path: Path):
    sr, x = wavfile.read(path)
    is_int = np.issubdtype(x.dtype, np.integer)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    if is_int:
        x = x / 32768.0        # int16 → [-1,1)
    return sr, x


def spec(x, sr):
    f, t, Z = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    mag = np.abs(Z)
    logmag = 20.0 * np.log10(mag + 1e-6)
    return f, t, mag, logmag


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def main():
    sr_n, npu = load(REF / "out_allnpu.wav")
    sr_f, fp = load(REF / "out_onnx_fp32.wav")
    assert sr_n == sr_f, f"sr mismatch {sr_n} != {sr_f}"
    sr = sr_n
    n = min(len(npu), len(fp)); npu, fp = npu[:n], fp[:n]

    f, t, mag_f, S_f = spec(fp, sr)
    _, _, mag_n, S_n = spec(npu, sr)

    wav_cos = cos(npu, fp)
    stft_cos = cos(mag_n, mag_f)
    rms_n, rms_f = float(np.sqrt((npu**2).mean())), float(np.sqrt((fp**2).mean()))

    vmin, vmax = -80.0, max(S_f.max(), S_n.max())
    extent = [t[0], t[-1], f[0] / 1000.0, f[-1] / 1000.0]
    kw = dict(origin="lower", aspect="auto", extent=extent, cmap="magma",
              vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ax[0].imshow(S_f, **kw)
    ax[0].set_title(f"fp32 (onnx)   RMS={rms_f:.4f}")
    im1 = ax[1].imshow(S_n, **kw)
    ax[1].set_title(f"NPU (LLM8850 U16)   RMS={rms_n:.4f}")
    for a in ax:
        a.set_xlabel("time [s]"); a.set_ylabel("freq [kHz]")
    fig.colorbar(im1, ax=ax[1], label="dB", fraction=0.046, pad=0.02)

    fig.suptitle(
        f"piper_plus_npu_opt  STFT — tokens.npz utterance (fp32 vs NPU U16)   "
        f"STFT-magnitude cos={stft_cos:.4f}",
        fontsize=12)

    out = REF / "stft_npu_vs_fp32.png"
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")
    print(f"  waveform cos = {wav_cos:.6f}")
    print(f"  STFT-mag cos = {stft_cos:.6f}")
    print(f"  RMS  npu={rms_n:.4f}  fp32={rms_f:.4f}  dur={n/sr:.2f}s")


if __name__ == "__main__":
    main()
