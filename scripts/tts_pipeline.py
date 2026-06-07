"""piper-plus 5-chunk 推論パイプライン (axengine NPU / onnxruntime CPU 両対応の共通 lib).

各 chunk (emb_lang / encp / dp / flow / decoder) を Session 抽象で叩き、間に CPU alignment
(duration → attn / y_mask) と math (z_p = m_p_e * y_mask、piper-plus 既定の noise-free) を挟んで
E2E 推論する。chunk の実行先は **拡張子で自動 dispatch**:

  *.axmodel  → axengine NPU (LLM8850 / AX650N NPU3, AX620E NPU2 等)
  *.onnx     → onnxruntime CPU (fp32 参照)

本セット (npu_opt) は **dp / flow を含む全 chunk を U16 で NPU 化済み**
(multi-stream iSTFT + dilation 6 への適応 FT で実現)。fp32 参照は同一 chunk を .onnx で
流すだけで得られる (run_tts_npu.py と run_tts_onnx.py は渡す paths の拡張子が違うだけ)。

主な API:
  load_tokens / Session / load_sessions / run_pipeline
  wav_stats / print_timing / print_quality / find_axmodel
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import cpu_alignment


# パイプライン全 chunk (report 表示順)
ALL_CHUNKS = ["emb_lang", "encp", "dp", "flow", "decoder"]

_ORT_DTYPE_MAP = {
    "tensor(float)":  np.float32, "tensor(float16)": np.float16,
    "tensor(int64)":  np.int64,   "tensor(int32)":   np.int32,
    "tensor(uint16)": np.uint16,  "tensor(int16)":   np.int16,
    "tensor(uint8)":  np.uint8,   "tensor(int8)":    np.int8,
}


# ─── Session 抽象 ────────────────────────────────────────────────────────

class Session:
    """`.onnx` → onnxruntime, `.axmodel` → axengine の自動 dispatch.

    axmodel 側は **pyaxengine** (`import axengine`, AXERA-TECH/pyaxengine) を使う。
    provider は axclrt (PCIe host) / axengine (SoC native) / auto。

    `run(feeds)` 時に入力 dtype を自動 cast し、session が受け取らない key も
    自動 filter する (axmodel/onnx 間で lid/x の int dtype が int32/int64 と
    違うことがあるため)。
    """

    def __init__(self, path: Path | str,
                 axe_provider: str | None = None, axe_device_id: int = 0):
        self.path = Path(path)
        self.backend = "axe" if self.path.suffix == ".axmodel" else "onnx"
        if self.backend == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.sess = ort.InferenceSession(
                str(self.path), opts, providers=["CPUExecutionProvider"])
        else:
            import axengine as axe  # pyaxengine (AXERA-TECH/pyaxengine)
            providers = None
            if axe_provider is not None and axe_provider != "auto":
                from axengine import axclrt_provider_name, axengine_provider_name
                m = {"axclrt": axclrt_provider_name, "axengine": axengine_provider_name}
                if axe_provider in m:
                    name = m[axe_provider]
                    providers = ([(name, {"device_id": axe_device_id})]
                                 if name == axclrt_provider_name else [name])
            self.sess = (axe.InferenceSession(str(self.path), providers=providers)
                         if providers else axe.InferenceSession(str(self.path)))
        self._dtypes = self._infer_dtypes()
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def _infer_dtypes(self) -> dict:
        out = {}
        for inp in self.sess.get_inputs():
            dt = getattr(inp, "dtype", None)
            if isinstance(dt, np.dtype):
                out[inp.name] = dt
                continue
            t = getattr(inp, "type", None)
            out[inp.name] = np.dtype(
                _ORT_DTYPE_MAP.get(t, np.float32) if isinstance(t, str) else np.float32
            )
        return out

    def run(self, feeds: dict) -> list[np.ndarray]:
        accepted = set(self.in_names)
        cast = {}
        for k, v in feeds.items():
            if k not in accepted:
                continue
            dt = self._dtypes.get(k)
            if dt is not None and v.dtype != dt:
                v = v.astype(dt)
            # pyaxengine の C API は ndarray の stride を無視して生バッファを読むため、
            # 非 C-contiguous な入力 (swapaxes/transpose 由来の view 等) を渡すと
            # 転置されたバイト列を NPU に流して破綻する (flow z_p で cos 0.14 ⇒ 0.999)。
            # axmodel backend では必ず C-contiguous 化する。
            if self.backend == "axe":
                v = np.ascontiguousarray(v)
            cast[k] = v
        return self.sess.run(None, cast)


def load_sessions(paths: dict[str, Path | str],
                  axe_provider: str | None = None, axe_device_id: int = 0,
                  ) -> dict[str, Session]:
    return {c: Session(p, axe_provider=axe_provider, axe_device_id=axe_device_id)
            for c, p in paths.items()}


# ─── tokens loader ────────────────────────────────────────────────────────

def load_tokens(npz_path: Path | str) -> tuple[dict, dict]:
    """tokens.npz を {phone/prosody_features/x_lengths/lid/g} + meta dict にロード."""
    npz = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    tokens = {k: npz[k] for k in
              ("phone", "prosody_features", "x_lengths", "lid", "g")}
    return tokens, meta


# ─── タイミング / 統計 helpers ───────────────────────────────────────────

def _time(fn, warmup: int, runs: int):
    out = None
    for _ in range(warmup):
        out = fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, ts


def stat(ts: list[float]) -> dict:
    a = np.asarray(ts, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()), n=len(a))


def fmt(s: dict) -> str:
    return (f"mean={s['mean']*1000:7.2f}ms  median={s['median']*1000:7.2f}ms  "
            f"min={s['min']*1000:7.2f}ms  max={s['max']*1000:7.2f}ms  n={s['n']}")


# ─── 音質 ─────────────────────────────────────────────────────────────────

def wav_stats(audio: np.ndarray, sr: int) -> dict:
    peak = int(np.abs(audio * 32767).max())
    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
    n_fft, hop = 1024, 512
    if len(audio) < n_fft:
        return dict(peak=peak, rms=rms, bands=[0.0] * 4)
    n_frames = (len(audio) - n_fft) // hop + 1
    bands = np.zeros(4, dtype=np.float64)
    win = np.hanning(n_fft)
    for f in range(n_frames):
        S = np.abs(np.fft.rfft(audio[f * hop : f * hop + n_fft] * win))
        bands += [S[:23].sum(), S[23:46].sum(), S[46:93].sum(), S[93:186].sum()]
    bands /= n_frames
    return dict(peak=peak, rms=rms, bands=bands.tolist())


# ─── パイプライン本体 ─────────────────────────────────────────────────────

def run_pipeline(sessions: dict[str, Session], tokens: dict, meta: dict,
                 params: dict, rng: np.random.Generator,
                 warmup: int = 5, runs: int = 20):
    """全 5 chunk + CPU alignment を順に走らせ、E2E audio と per-chunk 時間を返す.

    chunk の backend (NPU/CPU) は sessions に渡した Session が決める (本 lib は非依存)。

    Args:
      sessions: {chunk: Session}  (emb_lang / encp / dp / flow / decoder)
      tokens:   {phone, prosody_features, x_lengths, lid, g}
      meta:     {max_ph, max_t, inter_channels, hop_length, ...}
      params:   {noise_scale, length_scale}  (piper-plus 既定は noise-free)
    """
    MAX_PH = meta["max_ph"]; MAX_T = meta["max_t"]
    HIDDEN = meta.get("inter_channels", 192)
    HOP = meta.get("hop_length", 256)
    lid = tokens["lid"]
    times: dict[str, list[float]] = {}

    # emb_lang(lid) → g
    emb_out, times["emb_lang"] = _time(
        lambda: sessions["emb_lang"].run({"lid": lid}), warmup, runs)
    g = emb_out[0].astype(np.float32)

    # encp(x, x_lengths, g) → x_hidden, m_p, logs_p, x_mask
    encp_feeds = dict(x=tokens["phone"], x_lengths=tokens["x_lengths"], g=g)
    encp_out, times["encp"] = _time(
        lambda: sessions["encp"].run(encp_feeds), warmup, runs)
    enc_map = dict(zip(sessions["encp"].out_names, encp_out))
    x_hidden = enc_map.get("x_hidden", encp_out[0])
    # logs_p も encp は出すが noise-free 推論では未使用 (m_p / x_mask のみ使う)
    m_p, x_mask = enc_map["m_p"], enc_map["x_mask"]

    # dp: logw (duration)
    dp_feeds = dict(
        x=x_hidden, x_mask=x_mask, g=g,
        prosody_features=tokens["prosody_features"], lid=lid,
    )
    dp_out, times["dp"] = _time(
        lambda: sessions["dp"].run(dp_feeds), warmup, runs)
    logw = dp_out[0]

    # CPU alignment (duration → attn / y_mask)
    attn, y_mask, _noise_pre, y_len = cpu_alignment.make_alignment_inputs(
        logw, x_mask,
        length_scale=params["length_scale"], noise_scale=params["noise_scale"],
        hidden=HIDDEN, max_t=MAX_T, rng=rng,
    )

    # m_p をフレーム展開。piper-plus 既定は noise-free なので z_p = m_p_e * y_mask
    # (logs_p は noise 注入時のみ必要 → 本パイプラインでは未使用)。
    m_p_e = np.swapaxes(attn @ np.swapaxes(m_p, 1, 2), 1, 2)
    y_mask = y_mask.astype(np.float32)
    z_p = (m_p_e * y_mask).astype(np.float32)

    # flow
    flow_feeds = dict(z_p=z_p, y_mask=y_mask, g=g)
    flow_out, times["flow"] = _time(
        lambda: sessions["flow"].run(flow_feeds), warmup, runs)
    z_masked = (flow_out[0] * y_mask).astype(np.float32)

    # decoder
    dec_in_name = sessions["decoder"].in_names[0]
    dec_feeds = {dec_in_name: z_masked, "g": g}
    dec_out, times["decoder"] = _time(
        lambda: sessions["decoder"].run(dec_feeds), warmup, runs)
    audio = dec_out[0][0, 0, : y_len * HOP].astype(np.float32)

    return audio, times, y_len


# ─── レポート出力 ────────────────────────────────────────────────────────

def print_timing(times: dict[str, list[float]], runs: int, warmup: int,
                  backend_label: str, dur: float) -> None:
    print()
    print("=" * 78)
    print(f"  {backend_label}  (runs={runs}, warmup={warmup})")
    print("=" * 78)
    print(f"{'chunk':<16}     stats")
    print("-" * 78)
    total = 0.0
    for c in ALL_CHUNKS:
        if c not in times: continue
        s = stat(times[c]); total += s["mean"]
        print(f"{c:<16}     {fmt(s)}")
    print("-" * 78)
    rtf = total / dur if dur > 0 else float('nan')
    print(f"{'TOTAL (mean)':<16}     {total*1000:7.2f} ms"
          f"   (audio {dur:.2f}s → RTF {rtf:.3f})")


def print_quality(label: str, ws: dict, dur: float) -> None:
    b = ws["bands"]
    print(f"{label:<10} dur={dur:5.2f}s  peak={ws['peak']:5d}  "
          f"RMS={ws['rms']:.4f}  "
          f"0-1k={b[0]:6.2f}  1-2k={b[1]:6.2f}  "
          f"2-4k={b[2]:6.2f}  4-8k={b[3]:6.2f}")


def find_axmodel(axmodel_dir: Path | str, chunk: str) -> Path:
    d = Path(axmodel_dir)
    # chunk は必ず `ax650`/`ax620e` 等が後続するので `{chunk}<区切り>` で一意に引ける。
    # 区切りは `-` でも `_` でも可 (`[-_]`)、prefix 側の区切りには依存しない。
    p = next(d.glob(f"*{chunk}[-_]*.axmodel"), None)
    if p is None:
        raise FileNotFoundError(f"axmodel not found for chunk={chunk!r} in {d}")
    return p
