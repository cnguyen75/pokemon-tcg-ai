"""Pure-numpy inference for the learned leaf value net.

Loads value_net.npz (mean, std, W1, b1, W2, b2, W3, b3) exported by
train/train_net.py and exposes predict(features) -> win probability.
No torch/sklearn at inference; numpy only. If numpy or the weights file
is unavailable, OK stays False and callers fall back to the handcrafted eval.
"""
import os

OK = False
_W = None

try:
    import numpy as _np
except Exception:
    _np = None


def load(agent_dir=None):
    """Load weights; returns True on success. Safe to call more than once."""
    global OK, _W
    if _np is None:
        return False
    if agent_dir is None:
        agent_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(agent_dir, "value_net.npz")
    try:
        z = _np.load(path)
        _W = {k: _np.asarray(z[k], dtype=_np.float32)
              for k in ("mean", "std", "W1", "b1", "W2", "b2", "W3", "b3")}
        # transpose once so the hot loop is x @ W
        _W["W1t"] = _W["W1"].T.copy()
        _W["W2t"] = _W["W2"].T.copy()
        _W["W3t"] = _W["W3"].T.copy()
        OK = True
    except Exception:
        OK = False
    return OK


def predict(feats):
    """feats: list[float] (len == mean.size). Returns win probability."""
    x = (_np.asarray(feats, dtype=_np.float32) - _W["mean"]) / _W["std"]
    h = x @ _W["W1t"] + _W["b1"]
    _np.maximum(h, 0.0, out=h)
    h = h @ _W["W2t"] + _W["b2"]
    _np.maximum(h, 0.0, out=h)
    z = (h @ _W["W3t"] + _W["b3"])[0]
    if z >= 0.0:
        return float(1.0 / (1.0 + _np.exp(-z)))
    e = _np.exp(z)
    return float(e / (1.0 + e))


def predict_batch(X):
    """X: 2d array-like of feature rows. Returns 1d array of probabilities."""
    x = (_np.asarray(X, dtype=_np.float32) - _W["mean"]) / _W["std"]
    h = _np.maximum(x @ _W["W1t"] + _W["b1"], 0.0)
    h = _np.maximum(h @ _W["W2t"] + _W["b2"], 0.0)
    z = (h @ _W["W3t"] + _W["b3"]).ravel()
    return 1.0 / (1.0 + _np.exp(-z))
