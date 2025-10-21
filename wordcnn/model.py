from common.xp import xp as np
from common.layers import Convolution, Relu, Pooling, SoftmaxWithLoss

class WordCNN:
    def __init__(self, num_classes, weight_scale=0.01):
        C1, C2, C3 = 32, 64, 128
        self.params = {
            'W1': weight_scale * np.random.randn(C1, 1, 3, 3),
            'b1': np.zeros(C1),
            'W2': weight_scale * np.random.randn(C2, C1, 3, 3),
            'b2': np.zeros(C2),
            'W3': weight_scale * np.random.randn(C3, C2, 3, 3),
            'b3': np.zeros(C3),
            'W4': weight_scale * np.random.randn(C3, num_classes),
            'b4': np.zeros(num_classes)
        }
        self.layers = [
            Convolution(self.params['W1'], self.params['b1'], stride=1, pad=1),
            Relu(),
            Pooling(2, 2, stride=2),
            Convolution(self.params['W2'], self.params['b2'], stride=1, pad=1),
            Relu(),
            Pooling(2, 2, stride=2),
            Convolution(self.params['W3'], self.params['b3'], stride=1, pad=1),
            Relu(),
        ]
        self.last_layer = SoftmaxWithLoss()
        self._last_conv_out = None


    def _forward_conv(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self._last_conv_out = x  # (B,C,H,W)
        return x

    def _gap(self, x):
        # Global Average Pooling: (B,C,H,W) -> (B,C)
        return x.mean(axis=(2,3))

    def predict(self, x):
        z = self._forward_conv(x)
        f = self._gap(z)
        scores = f.dot(self.params['W4']) + self.params['b4']
        return scores

    def loss(self, x, t):
        x = np.asarray(x); t = np.asarray(t)
        scores = self.predict(x)
        return self.last_layer.forward(scores, t)

    def accuracy(self, x, t, topk=1):
        s = self.predict(x)
        if t.ndim != 1: t = t.argmax(axis=1)
        if topk == 1:
            y = s.argmax(axis=1)
            return float((y == t).mean())
        idx = np.argsort(s, axis=1)[:, -topk:]
        return float(np.mean([t[i] in idx[i] for i in range(len(t))]))

    def gradient(self, x, t):
        x = np.asarray(x); t = np.asarray(t)
        # forward
        z = self._forward_conv(x)        # (B,C,H,W)
        B, C, H, W = z.shape
        f = self._gap(z)                 # (B,C)
        scores = f.dot(self.params['W4']) + self.params['b4']
        loss = self.last_layer.forward(scores, t)

        # backward - linear
        ds = self.last_layer.backward()  # (B,num_classes)
        dW4 = f.T.dot(ds)                # (C,B)*(B,N)
        db4 = ds.sum(axis=0)
        df  = ds.dot(self.params['W4'].T)  # (B,C)

        # backward - GAP: 평균 역전파 = 균등 분배
        dz = np.ones_like(z) * (df[:, :, None, None] / (H * W))  # (B,C,H,W)

        # backward - conv stack (역순)
        for layer in self.layers[::-1]:
            dz = layer.backward(dz)

        grads = {
            'W1': self.layers[0].dW, 'b1': self.layers[0].db,
            'W2': self.layers[3].dW, 'b2': self.layers[3].db,
            'W3': self.layers[6].dW, 'b3': self.layers[6].db,
            'W4': dW4,                'b4': db4
        }
        return grads
