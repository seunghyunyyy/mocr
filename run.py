import argparse, os, subprocess, sys, time, json

def r(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main(args):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    ds_root = os.path.join(ROOT, args.dataset_root)              # dataset/
    art_root = os.path.join(ROOT, "artifacts")
    vocab_dir = os.path.join(art_root, "vocab")
    npy_dir   = os.path.join(art_root, "npy")
    ckpt_dir  = os.path.join(art_root, "ckpt")
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(npy_dir,   exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)

    t0 = time.time()

    # 1) Vocab
    if not args.skip_vocab:
        r([py, "-m", "util.build_vocab",
           "--json_dir", os.path.join(ds_root, "train"),
           "--out_dir", vocab_dir,
           "--coverage", str(args.coverage)] + (
              ["--max_vocab", str(args.max_vocab)] if args.max_vocab else []
          )
          )
    w2i = os.path.join(vocab_dir, "word2id.json")
    i2w = os.path.join(vocab_dir, "id2word.json")
    if not (os.path.exists(w2i) and os.path.exists(i2w)):
        raise SystemExit("Vocab 파일이 없습니다. build_vocab 단계가 실패했을 수 있어요.")

    # 2) Numpy 변환
    if not args.skip_numpy:
        r([py, "-m", "util.prepare_numpy",
           "--img_root", ds_root,
           "--vocab", w2i,
           "--out_dir", npy_dir,
           "--h", str(args.h),
           "--wmax", str(args.wmax),
           "--val_ratio", str(args.val_ratio)]
          )

    train_X = os.path.join(npy_dir, "train_X.npy")
    train_y = os.path.join(npy_dir, "train_y.npy")
    val_X   = os.path.join(npy_dir, "val_X.npy")
    val_y   = os.path.join(npy_dir, "val_y.npy")
    test_X  = os.path.join(npy_dir, "test_X.npy")
    test_y  = os.path.join(npy_dir, "test_y.npy")

    # 3) 학습
    if not args.skip_train:
        r([py, "-m", "wordcnn.train",
           "--train_x", train_X, "--train_y", train_y,
           "--val_x",   val_X,   "--val_y",   val_y,
           "--epochs",  str(args.epochs),
           "--batch",   str(args.batch),
           "--lr",      str(args.lr),
           "--ckpt_dir", ckpt_dir]
          )

    ckpt = os.path.join(ckpt_dir, "wordcnn_best.npz")
    if not os.path.exists(ckpt):
        raise SystemExit("학습 가중치가 없습니다. train 단계가 실패했을 수 있어요.")

    # 4) 평가
    if not args.skip_eval:
        r([py, "-m", "wordcnn.eval",
           "--test_x", test_X, "--test_y", test_y,
           "--ckpt", ckpt, "--id2word", i2w]
          )

    # 5) 추론(폴더 일괄 → CSV)
    if not args.skip_infer:
        out_csv = os.path.join(art_root, "preds_test.csv")
        r([py, "-m", "wordcnn.infer",
           "--img_dir", os.path.join(ds_root, "test"),
           "--ckpt", ckpt,
           "--id2word", i2w,
           "--out", out_csv,
           "--h", str(args.h),
           "--wmax", str(args.wmax),
           "--reject_tau", str(args.reject_tau)]
          )
        print("[DONE] CSV:", out_csv)

    print(f"\n[✅ ALL DONE] elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="dataset")
    ap.add_argument("--coverage", type=float, default=0.92)
    ap.add_argument("--max_vocab", type=int, default=None)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--wmax", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reject_tau", type=float, default=0.5)
    # Skips
    ap.add_argument("--skip_vocab", action="store_true")
    ap.add_argument("--skip_numpy", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval",  action="store_true")
    ap.add_argument("--skip_infer", action="store_true")
    args = ap.parse_args()
    main(args)
