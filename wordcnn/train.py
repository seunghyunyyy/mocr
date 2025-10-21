import argparse, os, numpy as onp
from common.xp import xp as xp
from wordcnn.model import WordCNN
from common.trainer import Trainer

def load_ckpt_into(net, ckpt_path):
    z = onp.load(ckpt_path, allow_pickle=True)
    # 1) params
    for k in net.params:
        if k in z.files:
            net.params[k] = xp.asarray(z[k])
    # 2) meta
    start_epoch = int(z["epoch"]) if "epoch" in z.files else 0
    best_val = float(z["best_val"]) if "best_val" in z.files else -1.0
    # 3) optimizer state (object array)
    opt_state = None
    if "opt" in z.files:
        arr = z["opt"]        # dtype=object, shape=(1,)
        opt_state = arr.item() if getattr(arr, "ndim", 0) > 0 else arr
    return start_epoch, best_val, opt_state

def main(args):
    Xtr = onp.load(args.train_x); ytr = onp.load(args.train_y)
    Xva = onp.load(args.val_x);   yva = onp.load(args.val_y)
    num_classes = int(ytr.max()) + 1

    net = WordCNN(num_classes=num_classes)

    start_epoch, best_val, opt_state = 0, -1.0, None
    if args.resume is not None:
        start_epoch, best_val, opt_state = load_ckpt_into(net, args.resume)
        print(f"[resume] loaded {args.resume} (start_epoch={start_epoch}, best={best_val:.4f})")

    trainer = Trainer(
        network=net,
        x_train=Xtr, t_train=ytr,
        x_test=Xva,  t_test=yva,
        epochs=args.epochs,
        mini_batch_size=args.batch,
        optimizer="adam", optimizer_param={"lr": args.lr},
        evaluate_sample_num_per_epoch=8192,
        eval_batch_size=1024,
        verbose=True,
        log_csv_path="artifacts/train_log.csv",
        start_epoch=start_epoch,
        ckpt_dir=args.ckpt_dir,
        ckpt_name="wordcnn",
        best_val_init=best_val,
    )

    # 옵티마이저 상태 복원 (네트워크/옵티마이저 초기화 직후)
    if opt_state is not None:
        trainer.load_optimizer_state(opt_state, net.params)
        print("[resume] optimizer state restored.")

    trainer.train()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_x", required=True); ap.add_argument("--train_y", required=True)
    ap.add_argument("--val_x",   required=True); ap.add_argument("--val_y",   required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--ckpt_dir", default="artifacts/ckpt")
    ap.add_argument("--resume", default=None, help="path to *.npz checkpoint")
    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)

