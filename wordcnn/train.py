import argparse, os, json, numpy as np
from common.trainer import Trainer
from common.optimizer import Adam
from wordcnn.model import WordCNN

def main(args):
    Xtr = np.load(args.train_x); ytr = np.load(args.train_y)
    Xva = np.load(args.val_x);   yva = np.load(args.val_y)
    num_classes = int(ytr.max()) + 1

    net = WordCNN(num_classes=num_classes)
    trainer = Trainer(net, Xtr, ytr, Xva, yva,
                      epochs=args.epochs, mini_batch_size=args.batch,
                      optimizer='adam', optimizer_param={'lr':args.lr},
                      verbose=True)
    trainer.train()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    np.savez(os.path.join(args.ckpt_dir, "wordcnn_best.npz"), **net.params)
    print("[train] saved:", os.path.join(args.ckpt_dir, "wordcnn_best.npz"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_x", required=True)
    ap.add_argument("--train_y", required=True)
    ap.add_argument("--val_x", required=True)
    ap.add_argument("--val_y", required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt_dir", default="artifacts/ckpt")
    args = ap.parse_args()
    main(args)
