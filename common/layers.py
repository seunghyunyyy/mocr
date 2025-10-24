# common/layers.py
# coding: utf-8
from collections import OrderedDict
from common.xp import xp as np, xp, DTYPE
from common.util import im2col, col2im

# ---------- softmax ----------
def _stable_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

# ---------- ReLU ----------
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        return dout

# ---------- Affine ----------
class Affine:
    def __init__(self, W, b):
        self.W = W; self.b = b
        self.x = None
        self.dW = None; self.db = None
    def forward(self, x):
        N = x.shape[0]
        self.x = x.reshape(N, -1)
        out = self.x @ self.W + self.b
        return out
    def backward(self, dout):
        dx = (dout @ self.W.T).reshape(self.x.shape)
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx

# ---------- Convolution ----------
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W.astype(DTYPE, copy=False)
        self.b = b.astype(DTYPE, copy=False)
        self.stride = int(stride); self.pad = int(pad)
        self.x = None; self.col = None; self.col_W = None
        self.dW = None; self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, Cx, H, W = x.shape
        assert Cx == C, f"Conv in_ch mismatch: x C={Cx} vs W C={C}"
        out_h = (H + 2*self.pad - FH)//self.stride + 1
        out_w = (W + 2*self.pad - FW)//self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)            # (N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T                           # (C*FH*FW, FN)
        out = col @ col_W + self.b                                 # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, FN).transpose(0,3,1,2)  # (N,FN,out_h,out_w)

        self.x = x; self.col = col; self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)  # (N*out_h*out_w, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = (self.col.T @ dout).T.reshape(self.W.shape)
        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

# ---------- Softmax with Loss ----------
class SoftmaxWithLoss:
    def __init__(self, label_smoothing=0.0):
        self.t = None; self.y = None; self.loss = None
        self.eps = float(label_smoothing)

    def forward(self, x, t):
        B = x.shape[0]
        self.t = t
        y = _stable_softmax(x.astype(np.float32, copy=False))  # 안정성 위해 확률은 float32
        self.y = y
        if t.ndim == 1:
            idx = t.astype(np.int64)
            if self.eps > 0:
                C = x.shape[1]
                q = np.full((B, C), self.eps/(C-1), dtype=np.float32)
                q[np.arange(B), idx] = 1.0 - self.eps
                self.loss = -np.mean(np.sum(q * np.log(y + 1e-12), axis=1))
            else:
                self.loss = -np.mean(np.log(y[np.arange(B), idx] + 1e-12))
        else:
            if self.eps > 0:
                C = x.shape[1]
                q = (1 - self.eps) * t + self.eps * (1 - t) / (C - 1)
            else:
                q = t
            self.loss = -np.mean(np.sum(q * np.log(y + 1e-12), axis=1))
        return np.asarray(self.loss, dtype=np.float32)

    def backward(self, dout=1):
        B = self.y.shape[0]
        if self.t.ndim == 1:
            idx = self.t.astype(np.int64)
            dx = self.y.copy()
            dx[np.arange(B), idx] -= 1.0
            dx /= B
        else:
            dx = (self.y - self.t) / B
        return dx.astype(DTYPE, copy=False)
