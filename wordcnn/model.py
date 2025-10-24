# wordcnn/model.py
# coding: utf-8
from collections import OrderedDict
from common.xp import xp, DTYPE
from common.layers import Convolution, Relu, Affine, SoftmaxWithLoss

class GlobalAvgPool:
    def __init__(self):
        self.x_shape = None
    def forward(self, x):
        self.x_shape = x.shape
        return xp.mean(x, axis=(2, 3))
    def backward(self, dout):
        B, C, H, W = self.x_shape
        return xp.broadcast_to(dout[:, :, None, None], (B, C, H, W)) / (H * W)

class WordCNN:
    """
    Conv(1→32) - ReLU
    Conv(32→64) - ReLU
    Conv(64→128) - ReLU
    GAP -> Affine(128→num_classes) -> SoftmaxWithLoss
    """
    def __init__(self, num_classes: int, weight_init_std: float = 0.01):
        C, F1, F2, F3 = 1, 32, 64, 128
        self.params = {}
        self.params['W1'] = xp.asarray(weight_init_std * xp.random.randn(F1, C, 3, 3), dtype=DTYPE)
        self.params['b1'] = xp.zeros(F1, dtype=DTYPE)
        self.params['W2'] = xp.asarray(weight_init_std * xp.random.randn(F2, F1, 3, 3), dtype=DTYPE)
        self.params['b2'] = xp.zeros(F2, dtype=DTYPE)
        self.params['W3'] = xp.asarray(weight_init_std * xp.random.randn(F3, F2, 3, 3), dtype=DTYPE)
        self.params['b3'] = xp.zeros(F3, dtype=DTYPE)
        self.params['W4'] = xp.asarray(weight_init_std * xp.random.randn(F3, num_classes), dtype=DTYPE)
        self.params['b4'] = xp.zeros(num_classes, dtype=DTYPE)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=1)
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride=1, pad=1)
        self.layers['Relu2'] = Relu()
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], stride=1, pad=1)
        self.layers['Relu3'] = Relu()
        self.layers['GAP']   = GlobalAvgPool()
        self.layers['Affine']= Affine(self.params['W4'], self.params['b4'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        z = x.astype(DTYPE, copy=False)
        for layer in self.layers.values():
            z = layer.forward(z)
        return z

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def _to_class_scores(self, scores):
        return scores - scores.max(axis=1, keepdims=True)

    def accuracy(self, x, t, topk=1):
        scores = self._to_class_scores(self.predict(x))
        if t.ndim == 2:
            t_idx = xp.argmax(t, axis=1)
        elif t.ndim == 1:
            t_idx = t.astype(xp.int64)
        else:
            raise ValueError(f"unexpected label shape: {t.shape}")

        if topk == 1:
            yhat = xp.argmax(scores, axis=1)
            return xp.mean((yhat == t_idx).astype(xp.float32))
        order = xp.argpartition(-scores, kth=topk-1, axis=1)[:, :topk]
        hits  = (order == t_idx[:, None]).any(axis=1)
        return xp.mean(hits.astype(xp.float32))

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Affine'].dW, self.layers['Affine'].db
        return grads

    # ── 평가 직전 캐시 해제용 ──
    def clear_state(self):
        for layer in self.layers.values():
            for attr in ("x", "col", "col_W", "mask", "y", "t"):
                if hasattr(layer, attr):
                    setattr(layer, attr, None)

    def cast_params_dtype(self, dtype: str = "auto"):
        dt = DTYPE if dtype == "auto" else (xp.float16 if "16" in dtype else xp.float32)
        for k in list(self.params.keys()):
            self.params[k] = self.params[k].astype(dt, copy=False)
