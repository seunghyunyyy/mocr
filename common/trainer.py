# common/trainer.py
# coding: utf-8
from typing import Optional, Dict, Any
import os, csv, time, atexit, signal
import numpy as onp

# tqdm(auto) + 폴백
try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:
        def __init__(self, it=None, **k): self.it = it or []
        def __iter__(self): return iter(self.it)
        def update(self, *a, **k): pass
        def set_postfix(self, **k): pass
        def close(self): pass
        @staticmethod
        def write(*a, **k): print(*a)

from common.xp import xp, asnumpy, DTYPE

# ---- Optimizers ---------------------------------------------------------
from common.optimizer import SGD, Adam

def _make_optimizer(name: str, opt: Dict[str, Any]):
    name = (name or "adam").lower()
    if name == "sgd":  return SGD(lr=opt.get("lr", 0.01))
    if name == "adam": return Adam(lr=opt.get("lr", 0.001),
                                   beta1=opt.get("beta1", 0.9),
                                   beta2=opt.get("beta2", 0.999))
    raise ValueError(f"Unknown optimizer: {name}")

# ---- Grad clip ----------------------------------------------------------
def _clip_grad_global_norm(grads: Dict[str, Any], max_norm: float = 0.0):
    if not max_norm or max_norm <= 0: return grads
    total = 0.0
    for g in grads.values(): total += float((asnumpy(g)**2).sum())
    total = (total ** 0.5) + 1e-12
    if total > max_norm:
        scale = max_norm / total
        for k in grads: grads[k] *= scale
    return grads

# ---- Trainer ------------------------------------------------------------
class Trainer:
    def __init__(
            self, network, x_train, t_train, x_test=None, t_test=None,
            epochs: int = 20, mini_batch_size: int = 64,
            optimizer: str = "adam", optimizer_param: Optional[Dict[str, Any]] = None,
            evaluate_sample_num_per_epoch: Optional[int] = 2048,
            eval_batch_size: int = 128, verbose: bool = True,
            log_csv_path: str = "artifacts/train_log.csv",
            # Resume/Save
            start_epoch: int = 0,
            ckpt_dir: Optional[str] = None, ckpt_name: str = "model",
            best_val_init: float = -1.0,
            # Speed/안정 옵션
            grad_clip: float = 5.0,
            warmup_epochs: int = 3,
            eval_every: int = 1,
            acc_steps: int = 1,
            # 도중 저장 옵션
            save_every_iters: int = 2000,
            save_every_secs: float = 300.0,
    ):
        self.network = network; self.verbose = verbose
        self.x_train = onp.asarray(x_train); self.t_train = onp.asarray(t_train)
        self.x_test  = None if x_test is None else onp.asarray(x_test)
        self.t_test  = None if t_test is None else onp.asarray(t_test)

        self.train_size = len(self.x_train)
        self.batch_size = int(mini_batch_size)
        self.epochs     = int(epochs)

        self.optimizer      = _make_optimizer(optimizer, optimizer_param or {})
        self._base_lr       = float(self.optimizer.lr)
        self.grad_clip      = float(grad_clip or 0.0)
        self.warmup_epochs  = int(warmup_epochs or 0)
        self.eval_every     = max(1, int(eval_every))
        self.acc_steps      = max(1, int(acc_steps))

        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.eval_batch_size = int(eval_batch_size)

        self.current_iter  = 0
        self.current_epoch = int(start_epoch)
        self.train_loss_list = []
        self.train_acc_list  = []
        self.test_acc_list   = []

        # 파라미터 dtype 정렬
        if hasattr(self.network, "params"):
            for k in list(self.network.params.keys()):
                self.network.params[k] = xp.asarray(self.network.params[k], dtype=DTYPE)

        # logging
        self.log_csv = log_csv_path
        log_dir = os.path.dirname(self.log_csv)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        with open(self.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","iter_global","iter_in_epoch","loss","train_acc","val_acc","time_s"])

        # ckpt
        self.ckpt_dir  = ckpt_dir
        self.ckpt_name = ckpt_name
        self.best_val  = float(best_val_init)
        if self.ckpt_dir: os.makedirs(self.ckpt_dir, exist_ok=True)

        # 도중 저장 주기
        self.save_every_iters = int(save_every_iters)
        self.save_every_secs  = float(save_every_secs)
        self._last_save_ts    = time.time()
        self._in_train        = False

        # 종료시 저장 등록
        self._register_savers()

    def _register_savers(self):
        def _atexit():
            if self._in_train:
                self._save_snapshot()
        atexit.register(_atexit)

        def _handler(signum, frame):
            tqdm.write(f"\n[signal {signum}] saving snapshot to '{self.ckpt_dir}' ...")
            self._save_snapshot()
            raise SystemExit(0)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception:
                pass

    def _sample_batch(self):
        idx = onp.random.choice(self.train_size, self.batch_size, replace=False)
        xb = xp.asarray(self.x_train[idx], dtype=DTYPE)
        yb = xp.asarray(self.t_train[idx])
        return xb, yb

    def _acc_mini_batches(self, Xh: onp.ndarray, Yh: onp.ndarray, topk=1) -> float:
        n = len(Yh)
        if n == 0: return 0.0
        hit, bs = 0.0, self.eval_batch_size
        for i in range(0, n, bs):
            xb = xp.asarray(Xh[i:i+bs], dtype=DTYPE); yb = xp.asarray(Yh[i:i+bs])
            acc = float(asnumpy(self.network.accuracy(xb, yb, topk=topk)))
            hit += acc * len(yb)
        return hit / n

    def _evaluate_epoch(self):
        # 평가 데이터 없거나, 샘플 수가 0/None이면 평가 생략
        if self.x_test is None or self.t_test is None:
            return None, None
        if not self.evaluate_sample_num_per_epoch or self.evaluate_sample_num_per_epoch <= 0:
            return None, None

        n = min(int(self.evaluate_sample_num_per_epoch), len(self.x_train))
        tr_idx = onp.random.choice(len(self.x_train), n, replace=False)
        te_idx = onp.random.choice(len(self.x_test),  min(n, len(self.x_test)), replace=False)
        xtr, ttr = self.x_train[tr_idx], self.t_train[tr_idx]
        xte, tte = self.x_test[te_idx],  self.t_test[te_idx]
        return self._acc_mini_batches(xtr, ttr, 1), self._acc_mini_batches(xte, tte, 1)

    def _save_epoch(self, epoch:int, val_acc):
        if not self.ckpt_dir: return
        payload = {k: asnumpy(v) for k,v in self.network.params.items()}
        payload["epoch"] = int(epoch)
        if val_acc is not None:
            payload["best_val"] = float(self.best_val)
        if hasattr(self.optimizer, "state_dict"):
            try:
                payload["opt"] = self.optimizer.state_dict()
            except Exception:
                pass
        onp.savez(os.path.join(self.ckpt_dir, f"{self.ckpt_name}_last.npz"), **payload)
        if val_acc is not None and val_acc > self.best_val:
            self.best_val = float(val_acc)
            onp.savez(os.path.join(self.ckpt_dir, f"{self.ckpt_name}_best.npz"), **payload)

    def _save_snapshot(self):
        try:
            self._save_epoch(self.current_epoch, val_acc=None)
        except Exception as e:
            tqdm.write(f"[warn] snapshot save failed: {e}")

    def _apply_warmup(self):
        if self.warmup_epochs <= 0: return
        if self.current_epoch < self.warmup_epochs:
            self.optimizer.lr = self._base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            self.optimizer.lr = self._base_lr

    def load_optimizer_state(self, state, params):
        if state is None: return
        if hasattr(self.optimizer, "load_state_dict"):
            try:
                self.optimizer.load_state_dict(state, params)
            except Exception:
                pass

    def _free_cupy_pools(self):
        try:
            import cupy as _cp
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def train(self):
        iters_per_epoch = max(self.train_size // self.batch_size, 1)
        start_t = time.time()

        ep_iter = range(self.current_epoch, self.epochs)
        ep_bar = tqdm(ep_iter, desc="epochs", dynamic_ncols=True, ascii=True, position=0, leave=True)

        self._in_train = True
        try:
            for ep in ep_bar:
                self._apply_warmup()

                it_bar = tqdm(range(iters_per_epoch),
                              desc=f"iters(ep={ep+1})", dynamic_ncols=True, ascii=True,
                              position=1, leave=False, unit="it", mininterval=0.5)

                last_loss, step_in_epoch = None, 0
                acc_count, acc_grads = 0, None

                for _ in it_bar:
                    xb, yb = self._sample_batch()
                    loss = self.network.loss(xb, yb)
                    grads = self.network.gradient(xb, yb)

                    if self.acc_steps > 1:
                        if acc_grads is None:
                            acc_grads = {k: grads[k].copy() for k in grads}
                        else:
                            for k in grads: acc_grads[k] += grads[k]
                        acc_count += 1
                        if acc_count == self.acc_steps:
                            for k in acc_grads: acc_grads[k] /= self.acc_steps
                            acc_grads = _clip_grad_global_norm(acc_grads, self.grad_clip)
                            self.optimizer.update(self.network.params, acc_grads)
                            acc_count, acc_grads = 0, None
                    else:
                        grads = _clip_grad_global_norm(grads, self.grad_clip)
                        self.optimizer.update(self.network.params, grads)

                    last_loss = float(asnumpy(loss))
                    self.train_loss_list.append(last_loss)
                    self.current_iter += 1
                    step_in_epoch += 1
                    it_bar.set_postfix(loss=f"{last_loss:.4f}")

                    # 주기적 스냅샷
                    now = time.time()
                    if (self.save_every_iters and self.current_iter % self.save_every_iters == 0) or \
                            (self.save_every_secs and now - self._last_save_ts >= self.save_every_secs):
                        self._save_snapshot()
                        self._last_save_ts = now

                # 누적 잔여 처리
                if self.acc_steps > 1 and acc_grads is not None:
                    for k in acc_grads: acc_grads[k] /= max(1, acc_count)
                    acc_grads = _clip_grad_global_norm(acc_grads, self.grad_clip)
                    self.optimizer.update(self.network.params, acc_grads)

                self.current_epoch = ep + 1

                # ── 평가 직전: 임시 버퍼 참조 해제 + CuPy 풀 비우기 ──
                if hasattr(self.network, "clear_state"):
                    self.network.clear_state()
                self._free_cupy_pools()

                # 평가
                if (self.current_epoch % self.eval_every) == 0:
                    tr_acc, val_acc = self._evaluate_epoch()
                else:
                    tr_acc, val_acc = None, None

                if tr_acc is not None:
                    self.train_acc_list.append(tr_acc); self.test_acc_list.append(val_acc)
                    tqdm.write(f"[epoch {self.current_epoch}/{self.epochs}] "
                               f"loss={last_loss:.4f}  train={tr_acc:.3f}  val={val_acc:.3f}  best={self.best_val:.3f}")
                else:
                    tqdm.write(f"[epoch {self.current_epoch}/{self.epochs}] loss={last_loss:.4f}")

                with open(self.log_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self.current_epoch, self.current_iter, step_in_epoch,
                        f"{(last_loss or 0.0):.6f}",
                        f"{(tr_acc or 0.0):.6f}", f"{(val_acc or 0.0):.6f}",
                        f"{time.time()-start_t:.1f}"
                    ])

                self._save_epoch(self.current_epoch, val_acc)
        finally:
            self._in_train = False
