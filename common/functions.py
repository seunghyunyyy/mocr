# common/functions.py
# coding: utf-8
import numpy as np

def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise 안정화된 softmax (log-sum-exp)."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy_from_logits(logits: np.ndarray, t: np.ndarray, eps: float = 0.1) -> float:
    """
    안정화된 softmax + 라벨 스무딩 CE.
    logits: (B, C)
    t: (B,) int labels  또는  (B, C) one-hot
    eps: 0.0~0.2 권장
    """
    p = stable_softmax(logits)  # (B, C)

    if t.ndim == 1:
        B, C = logits.shape
        if eps > 0:
            q = np.full((B, C), eps/(C-1), dtype=logits.dtype)
            q[np.arange(B), t] = 1.0 - eps
            loss = -(q * np.log(p + 1e-12)).sum(axis=1).mean()
        else:
            loss = -np.log(p[np.arange(B), t] + 1e-12).mean()
    elif t.ndim == 2:
        C = logits.shape[1]
        if eps > 0:
            q = (1 - eps) * t + eps * (1 - t) / (C - 1)
        else:
            q = t
        loss = -(q * np.log(p + 1e-12)).sum(axis=1).mean()
    else:
        raise ValueError(f"Unexpected label shape: {t.shape}")
    return float(loss)