from typing import Dict, Any
from common.xp import xp as xp, asnumpy

class SGD:
    def __init__(self, lr=0.01):
        self.lr = float(lr)
        self.state = {}  # SGD는 특별한 모멘트 없음(호환용 빈 dict)

    def update(self, params: Dict[str, xp.ndarray], grads: Dict[str, xp.ndarray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

    # ----- state I/O -----
    def state_dict(self, params: Dict[str, xp.ndarray]) -> Dict[str, Any]:
        return {
            "type": "sgd",
            "hyper": {"lr": self.lr},
            "state": {},  # 빈 상태
        }

    def load_state_dict(self, sd: Dict[str, Any], params: Dict[str, xp.ndarray]):
        # 하이퍼만 복구
        if sd and sd.get("type") == "sgd":
            self.lr = float(sd.get("hyper", {}).get("lr", self.lr))


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.iter = 0
        self.m = {}  # 1차 모멘트
        self.v = {}  # 2차 모멘트

    def update(self, params: Dict[str, xp.ndarray], grads: Dict[str, xp.ndarray]):
        self.iter += 1
        lr_t = self.lr * xp.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for key in params.keys():
            if key not in self.m:
                self.m[key] = xp.zeros_like(params[key])
                self.v[key] = xp.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grads[key] ** 2)
            params[key] -= lr_t * self.m[key] / (xp.sqrt(self.v[key]) + 1e-7)

    # ----- state I/O -----
    def state_dict(self, params: Dict[str, xp.ndarray]) -> Dict[str, Any]:
        # CuPy/NumPy 상관없이 넘파이로 저장
        m_np = {k: asnumpy(v) for k, v in self.m.items()}
        v_np = {k: asnumpy(v) for k, v in self.v.items()}
        return {
            "type": "adam",
            "hyper": {"lr": self.lr, "beta1": self.beta1, "beta2": self.beta2},
            "iter": int(self.iter),
            "m": m_np,
            "v": v_np,
        }

    def load_state_dict(self, sd: Dict[str, Any], params: Dict[str, xp.ndarray]):
        if not sd or sd.get("type") != "adam":
            return
        h = sd.get("hyper", {})
        self.lr = float(h.get("lr", self.lr))
        self.beta1 = float(h.get("beta1", self.beta1))
        self.beta2 = float(h.get("beta2", self.beta2))
        self.iter = int(sd.get("iter", 0))
        # 사전 키 기준으로만 복원 (없으면 자동 초기화되게 둠)
        self.m = {k: xp.asarray(v) for k, v in sd.get("m", {}).items() if k in params}
        self.v = {k: xp.asarray(v) for k, v in sd.get("v", {}).items() if k in params}

