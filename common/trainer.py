from typing import Optional, Dict, Any
from tqdm import trange, tqdm
import os, csv, time, numpy as onp
from common.xp import xp as xp, asnumpy
from common.optimizer import SGD, Adam

def _make_optimizer(name: str, opt: Dict[str, Any]):
    name = (name or "sgd").lower()
    if name == "sgd":  return SGD(lr=opt.get("lr", 0.01))
    if name == "adam": return Adam(lr=opt.get("lr", 0.001),
                                   beta1=opt.get("beta1", 0.9),
                                   beta2=opt.get("beta2", 0.999))
    raise ValueError(f"Unknown optimizer: {name}")

class Trainer:
    def __init__(
        self,
        network, x_train, t_train, x_test=None, t_test=None,
        epochs: int = 20, mini_batch_size: int = 100,
        optimizer: str = "sgd", optimizer_param: Optional[Dict[str, Any]] = None,
        evaluate_sample_num_per_epoch: Optional[int] = 8192,
        eval_batch_size: int = 1024, verbose: bool = True,
        log_csv_path: str = "artifacts/train_log.csv",
        # resume/save
        start_epoch: int = 0,
        ckpt_dir: Optional[str] = None,
        ckpt_name: str = "model",
        best_val_init: float = -1.0,
    ):
        self.network = network; self.verbose = verbose
        self.x_train = onp.asarray(x_train); self.t_train = onp.asarray(t_train)
        self.x_test  = None if x_test is None else onp.asarray(x_test)
        self.t_test  = None if t_test is None else onp.asarray(t_test)

        self.train_size = len(self.x_train)
        self.batch_size = int(mini_batch_size)
        self.epochs     = int(epochs)

        self.optimizer  = _make_optimizer(optimizer, optimizer_param or {})
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.eval_batch_size = int(eval_batch_size)

        self.current_iter  = 0
        self.current_epoch = int(start_epoch)
        self.train_loss_list = []
        self.train_acc_list  = []
        self.test_acc_list   = []

        # logging
        self.log_csv = log_csv_path
        log_dir = os.path.dirname(self.log_csv)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        with open(self.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","iter_global","iter_in_epoch","loss","train_acc","val_acc","time_s"])

        # checkpointing
        self.ckpt_dir  = ckpt_dir
        self.ckpt_name = ckpt_name
        self.best_val  = float(best_val_init)
        if self.ckpt_dir: os.makedirs(self.ckpt_dir, exist_ok=True)

    # --------- public: resume optimizer ----------
    def load_optimizer_state(self, state: Dict[str, Any], params: Dict[str, xp.ndarray]):
        if not state: return
        typ = state.get("type", "")
        self.optimizer.load_state_dict(state, params)

    # --------- I/O ----------
    def _save_epoch(self, epoch:int, val_acc: Optional[float]):
        if not self.ckpt_dir: return
        payload = {k: asnumpy(v) for k, v in self.network.params.items()}
        payload["epoch"] = int(epoch)
        payload["best_val"] = float(self.best_val if val_acc is not None else self.best_val)
        # optimizer state를 object 배열로 저장 (np.load(..., allow_pickle=True) 필요)
        opt_state = self.optimizer.state_dict(self.network.params)
        payload["opt"] = onp.array([opt_state], dtype=object)

        last = os.path.join(self.ckpt_dir, f"{self.ckpt_name}_last.npz")
        onp.savez(last, **payload)
        if val_acc is not None and val_acc > self.best_val:
            self.best_val = float(val_acc)
            best = os.path.join(self.ckpt_dir, f"{self.ckpt_name}_best.npz")
            onp.savez(best, **payload)

    # --------- helpers ----------
    def _sample_batch(self):
        idx = onp.random.choice(self.train_size, self.batch_size, replace=False)
        xb = xp.asarray(self.x_train[idx]); yb = xp.asarray(self.t_train[idx])
        return xb, yb

    def _acc_mini_batches(self, Xh: onp.ndarray, Yh: onp.ndarray, topk=1) -> float:
        n = len(Yh)
        if n == 0: return 0.0
        hit, bs = 0.0, self.eval_batch_size
        for i in range(0, n, bs):
            xb = xp.asarray(Xh[i:i+bs]); yb = xp.asarray(Yh[i:i+bs])
            acc = float(asnumpy(self.network.accuracy(xb, yb, topk=topk)))
            hit += acc * len(yb)
        return hit / n

    def _evaluate_epoch(self):
        if self.x_test is None or self.t_test is None: return None, None
        if self.evaluate_sample_num_per_epoch is None:
            xtr, ttr = self.x_train, self.t_train
            xte, tte = self.x_test,  self.t_test
        else:
            n = min(int(self.evaluate_sample_num_per_epoch), len(self.x_train))
            tr_idx = onp.random.choice(len(self.x_train), n, replace=False)
            te_idx = onp.random.choice(len(self.x_test),  min(n, len(self.x_test)), replace=False)
            xtr, ttr = self.x_train[tr_idx], self.t_train[tr_idx]
            xte, tte = self.x_test[te_idx],  self.t_test[te_idx]
        return self._acc_mini_batches(xtr, ttr, 1), self._acc_mini_batches(xte, tte, 1)

    # --------- main ----------
    def train(self, save_every: Optional[int] = None):
        iters_per_epoch = max(self.train_size // self.batch_size, 1)
        start_t = time.time()

        try:
            for ep in trange(self.current_epoch, self.epochs, desc="epochs"):
                it_bar = trange(iters_per_epoch, desc="iters", leave=False, unit="it")
                for it_in_ep in it_bar:
                    xb, yb = self._sample_batch()
                    loss = self.network.loss(xb, yb)
                    grads = self.network.gradient(xb, yb)
                    self.optimizer.update(self.network.params, grads)
                    self.train_loss_list.append(float(asnumpy(loss)))
                    self.current_iter += 1
                    it_bar.set_postfix(loss=f"{self.train_loss_list[-1]:.4f}")

                    if save_every and self.ckpt_dir and (self.current_iter % int(save_every) == 0):
                        self._save_epoch(ep, val_acc=None)

                # epoch end
                self.current_epoch = ep + 1
                tr_acc, val_acc = self._evaluate_epoch()
                last_loss = self.train_loss_list[-1]

                if tr_acc is not None:
                    self.train_acc_list.append(tr_acc); self.test_acc_list.append(val_acc)
                    tqdm.write(f"[epoch {self.current_epoch}/{self.epochs}] "
                               f"loss={last_loss:.4f}  train={tr_acc:.3f}  val={val_acc:.3f}  best={self.best_val:.3f}")
                else:
                    tqdm.write(f"[epoch {self.current_epoch}/{self.epochs}] loss={last_loss:.4f}")

                with open(self.log_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self.current_epoch, self.current_iter, iters_per_epoch,
                        f"{last_loss:.6f}", f"{(tr_acc or 0.0):0.6f}",
                        f"{(val_acc or 0.0):0.6f}", f"{time.time()-start_t:.1f}"
                    ])
                self._save_epoch(self.current_epoch, val_acc)

        except KeyboardInterrupt:
            tqdm.write("[trainer] Interrupted. Saving last checkpoint...")
            self._save_epoch(self.current_epoch, val_acc=None)
            raise

