# wordcnn/train.py
# coding: utf-8
"""
- 체크포인트 in-place 로드(필수): net.params[k][...] = xp.asarray(z[k])
- --resume 주면 ckpt의 W4.shape[1]로 num_classes를 강제하여 shape mismatch 방지
- 진행 로그, 주기적 저장, 주기적 평가 지원
"""
import argparse, os, numpy as onp
from common.xp import xp, asnumpy, backend_name
from wordcnn.model import WordCNN
from common.trainer import Trainer


def _infer_num_classes(y_train_path: str) -> int:
    ytr = onp.load(y_train_path, mmap_mode=None)
    return int(ytr.max()) + 1 if len(ytr) > 0 else 1


def _num_classes_from_ckpt(ckpt_path: str) -> int:
    z = onp.load(ckpt_path, allow_pickle=True)
    if "W4" not in z.files:
        raise ValueError(f"[resume] ckpt '{ckpt_path}' has no 'W4' param to infer out_dim.")
    return int(z["W4"].shape[1])


def load_ckpt_into(net, ckpt_path):
    """가중치/메타/옵티마이저 상태 로드 (모두 선택). 가중치는 반드시 in-place 복사!"""
    z = onp.load(ckpt_path, allow_pickle=True)
    loaded, mism = 0, []
    for k in net.params:
        if k in z.files:
            w = xp.asarray(z[k])
            if w.shape == net.params[k].shape:
                net.params[k][...] = w       # ★ in-place 복사 (참조 유지)
                loaded += 1
            else:
                mism.append((k, w.shape, net.params[k].shape))
    start_epoch = int(z["epoch"]) if "epoch" in z.files else 0
    best_val    = float(z["best_val"]) if "best_val" in z.files else -1.0

    opt_state = None
    if "opt" in z.files:
        arr = z["opt"]
        opt_state = arr.item() if getattr(arr, "ndim", 0) > 0 else arr

    if mism:
        print("[warn] shape mismatch (ckpt vs net):", mism)
    print(f"[resume] loaded params: {loaded}/{len(net.params)}")
    return start_epoch, best_val, opt_state


def main(args):
    # 백엔드/DTYPE 정보
    print(f"[backend] {backend_name()}  dtype={xp.asarray(0.0).dtype}")

    # 데이터 로드
    Xtr = onp.load(args.train_x)  # (N,1,H,W) float [-1,1]
    ytr = onp.load(args.train_y)  # (N,) int
    Xva = onp.load(args.val_x)
    yva = onp.load(args.val_y)

    # 클래스 수 결정: 기본은 train 라벨, --resume이면 ckpt의 out_dim으로 강제
    num_classes = _infer_num_classes(args.train_y)
    if args.resume:
        try:
            ckpt_out = _num_classes_from_ckpt(args.resume)
            num_classes = ckpt_out
            print(f"[resume] override num_classes from ckpt: {num_classes}")
        except Exception as e:
            print(f"[resume] failed to infer num_classes from ckpt: {e}")
            print(f"[resume] fallback to train labels: {num_classes}")

    # 안전장치: 라벨 범위 점검
    if int(ytr.max()) >= num_classes:
        raise ValueError(f"[error] train_y max label {int(ytr.max())} >= num_classes {num_classes}")

    # 네트워크 생성
    net = WordCNN(num_classes=num_classes)

    # 체크포인트 로드 (선택)
    start_epoch, best_val, opt_state = 0, -1.0, None
    if args.resume:
        start_epoch, best_val, opt_state = load_ckpt_into(net, args.resume)
        print(f"[resume] loaded {args.resume} (start_epoch={start_epoch}, best={best_val:.4f})")

    # Trainer 구성
    eval_samples = None if (args.eval_samples is None or args.eval_samples <= 0) else args.eval_samples
    trainer = Trainer(
        network=net,
        x_train=Xtr, t_train=ytr,
        x_test=Xva,  t_test=yva,
        epochs=args.epochs,
        mini_batch_size=args.batch,
        optimizer="adam", optimizer_param={"lr": args.lr},
        evaluate_sample_num_per_epoch=eval_samples,
        eval_batch_size=args.eval_bs,
        verbose=True,
        log_csv_path="artifacts/train_log.csv",
        start_epoch=start_epoch,
        ckpt_dir=args.ckpt_dir,
        ckpt_name="wordcnn",
        best_val_init=best_val,
        # 아래 옵션들은 common/trainer.py에 구현돼 있어야 합니다.
        grad_clip=5.0,            # 그라디언트 클립
        warmup_epochs=3,          # 초반 워밍업(Trainer가 지원하는 경우)
        eval_every=max(1, args.eval_every),
        acc_steps=max(1, args.acc_steps),   # gradient accumulation
        save_every_iters=args.save_every_iters,
        save_every_secs=args.save_every_secs,
    )

    # 옵티마이저 상태 복원(있다면)
    if opt_state is not None:
        try:
            trainer.load_optimizer_state(opt_state, net.params)
            print("[resume] optimizer state restored.")
        except Exception as e:
            print(f"[resume] optimizer state restore failed (continue w/ fresh opt): {e}")

    # 학습 시작
    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_x", required=True)
    ap.add_argument("--train_y", required=True)
    ap.add_argument("--val_x",   required=True)
    ap.add_argument("--val_y",   required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--ckpt_dir", default="artifacts/ckpt")
    ap.add_argument("--resume", default=None, help="path to *.npz checkpoint")
    ap.add_argument("--acc_steps", type=int, default=1)
    # 평가 옵션
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--eval_samples", type=int, default=1024, help="0 or None = skip eval per epoch")
    ap.add_argument("--eval_every", type=int, default=1)
    # 도중 저장 주기
    ap.add_argument("--save_every_iters", type=int, default=2000)
    ap.add_argument("--save_every_secs", type=float, default=300.0)

    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)