"""CPU alignment helper (numpy 実装).

piper-plus の `dp` (duration predictor) 出力 logw から、`flow` / decoder に渡す
alignment (attn) と frame マスク (y_mask) を生成する。NN ではない決定的処理なので
NPU には載せず CPU で実行する (VITS 共通)。AND ベースの path 構成を numpy 化したもの。

  入力:
    logw          [1, 1, P=MAX_PH]   (dp 出力)
    x_mask        [1, 1, P]          (encp 出力、phoneme 有効領域マスク)
    length_scale  float              (発話速度、推論時パラメータ)
    noise_scale   float              (任意の noise 注入用。piper-plus 既定は noise-free)

  出力:
    attn          [1, T=MAX_T, P]    (alignment matrix、y_mask + x_mask 適用済)
    y_mask        [1, 1, T]
    noise_pre     [1, hidden, T]     (= randn * noise_scale。noise-free 推論では未使用)
    y_lengths     int                (実際の audio frame 数)

  使い方 (run_pipeline 内):
    attn, y_mask, _noise_pre, y_len = make_alignment_inputs(
        logw, x_mask, length_scale=1.0, noise_scale=0.667, hidden=192, max_t=512)
    # m_p_e = attn @ m_p ;  z_p = m_p_e * y_mask  → flow → decoder
"""
from __future__ import annotations

import numpy as np

MAX_PH_DEFAULT = 512
MAX_T_DEFAULT = 512
HIDDEN_DEFAULT = 192


def make_alignment(
    logw: np.ndarray,
    x_mask: np.ndarray,
    length_scale: float = 1.0,
    max_t: int = MAX_T_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, int]:
    """logw + x_mask から (attn, y_mask, y_lengths) を生成.

    Args:
      logw:   [1, 1, P]  float32  (dp 出力)
      x_mask: [1, 1, P]  float32  (encp 出力、phoneme 有効領域マスク)
      length_scale: 発話速度スケール
      max_t: 出力 frame 数上限 (固定形状なので MAX_T 一律)

    Returns:
      attn:      [1, T=max_t, P]  float32  (alignment matrix、y_mask + x_mask 適用済)
      y_mask:    [1, 1, T]        float32  (frame 有効領域マスク)
      y_lengths: int                       (実際の有効 frame 数 ≤ max_t)
    """
    assert logw.ndim == 3 and x_mask.ndim == 3
    P = logw.shape[-1]
    T = max_t

    # 1) duration  w = exp(logw) * x_mask * length_scale
    w = np.exp(logw) * x_mask * length_scale            # [1, 1, P]
    w_ceil = np.ceil(w).astype(np.float32)              # [1, 1, P]

    # 2) y_lengths = clamp(sum(w_ceil), 1, T)
    y_lengths = max(1, min(T, int(w_ceil.sum())))

    # 3) y_mask [1, 1, T]
    arange_t_1d = np.arange(T, dtype=np.float32)         # [T]
    y_mask = (arange_t_1d < y_lengths).astype(np.float32).reshape(1, 1, T)

    # 4) attn_mask [1, 1, T, P]
    attn_mask = y_mask[..., None] * x_mask[:, :, None, :]

    # 5) cum_duration / cum_dur_prev
    cum_dur = np.cumsum(w_ceil, axis=2)                  # [1, 1, P]
    cum_dur_prev = np.pad(cum_dur[:, :, :-1], ((0, 0), (0, 0), (1, 0)))

    # 6) path via AND
    arange_t_4d = arange_t_1d.reshape(1, 1, 1, T)        # [1, 1, 1, T]
    mask_lt  = arange_t_4d < cum_dur[..., None]          # [1, 1, P, T]
    mask_gte = arange_t_4d >= cum_dur_prev[..., None]    # [1, 1, P, T]
    path = (mask_lt & mask_gte).astype(np.float32)       # [1, 1, P, T]

    # 7) transpose to [1, 1, T, P] + apply attn_mask
    attn = path.transpose(0, 1, 3, 2) * attn_mask        # [1, 1, T, P]

    return attn.squeeze(1).astype(np.float32), y_mask.astype(np.float32), y_lengths


def make_noise_pre(
    shape: tuple[int, int, int],
    noise_scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """noise_pre = randn(shape) * noise_scale を生成 (任意の noise 注入用)."""
    if rng is None:
        rng = np.random.default_rng()
    return (rng.standard_normal(shape) * noise_scale).astype(np.float32)


def make_alignment_inputs(
    logw: np.ndarray,
    x_mask: np.ndarray,
    length_scale: float = 1.0,
    noise_scale: float = 0.6,
    hidden: int = HIDDEN_DEFAULT,
    max_t: int = MAX_T_DEFAULT,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """logw + x_mask から alignment 入力 (attn, y_mask, noise_pre, y_len) を一括生成.

    piper-plus 既定の noise-free 推論では noise_pre は使わない (attn / y_mask のみ使用)。
    """
    attn, y_mask, y_lengths = make_alignment(logw, x_mask, length_scale=length_scale, max_t=max_t)
    noise_pre = make_noise_pre((1, hidden, max_t), noise_scale=noise_scale, rng=rng)
    return attn, y_mask, noise_pre, y_lengths


if __name__ == "__main__":
    # 自己テスト
    P = MAX_PH_DEFAULT
    T = MAX_T_DEFAULT
    rng = np.random.default_rng(42)

    # 簡易ダミー: 100 phonemes 有効、各 ~5 frames
    PHONE_LEN = 100
    x_mask = np.zeros((1, 1, P), dtype=np.float32)
    x_mask[..., :PHONE_LEN] = 1.0
    logw = np.zeros((1, 1, P), dtype=np.float32)
    logw[..., :PHONE_LEN] = np.log(5.0)                  # 各 phone 5 frames

    attn, y_mask, noise_pre, y_len = make_alignment_inputs(
        logw, x_mask, length_scale=1.0, noise_scale=0.6, rng=rng,
    )
    print(f"y_lengths = {y_len}")
    print(f"attn shape={attn.shape} sum={attn.sum():.1f} (expected ~{y_len})")
    print(f"y_mask shape={y_mask.shape} sum={y_mask.sum():.0f} (expected {y_len})")
    print(f"noise_pre shape={noise_pre.shape} mean={noise_pre.mean():.3f} std={noise_pre.std():.3f}")
    # 各 frame は 1 つの phoneme に対応 → attn.sum(axis=-1) は y_mask
    per_frame_sum = attn.sum(axis=-1)  # [1, T]
    print(f"per-frame attn sum: min={per_frame_sum.min():.1f} max={per_frame_sum.max():.1f}")
    # 各 phoneme の duration
    per_phone_sum = attn.sum(axis=-2)  # [1, P]
    print(f"per-phone attn sum (first 5): {per_phone_sum[0, :5]}")
