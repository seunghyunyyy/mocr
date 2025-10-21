import argparse, os, glob, json, math
import numpy as np
from PIL import Image
import unicodedata

def normalize_token(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())

def crop_from_quad(img: Image.Image, xs, ys, pad=2):
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(img.width,  x2 + pad); y2 = min(img.height, y2 + pad)
    return img.crop((x1, y1, x2, y2))

def resize_h_pad_w(img: Image.Image, H=32, Wmax=256):
    w, h = img.size
    new_w = max(1, int(round(w * (H / h))))
    img = img.resize((new_w, H), Image.BILINEAR)
    # 우측 제로패딩
    if new_w > Wmax:
        img = img.resize((Wmax, H), Image.BILINEAR)
        new_w = Wmax
    canvas = Image.new("L", (Wmax, H), color=0)
    canvas.paste(img, (0, 0))
    return canvas

def process_split(split_dir, word2id, out_x, out_y, H=32, Wmax=256):
    img_paths = sorted(glob.glob(os.path.join(split_dir, "*.png")))
    X, y = [], []
    for ip in img_paths:
        jp = ip[:-4] + ".json"
        if not os.path.exists(jp): continue
        img = Image.open(ip).convert("L")
        with open(jp, "r", encoding="utf-8") as f:
            j = json.load(f)
        for b in j.get("bbox", []):
            word = normalize_token(str(b.get("data", "")))
            if word not in word2id:
                continue  # 폐어휘: 상위 N에 없는 단어 제외
            xs, ys = b.get("x", []), b.get("y", [])
            if len(xs) != 4 or len(ys) != 4:
                continue
            crop = crop_from_quad(img, xs, ys, pad=2)
            patch = resize_h_pad_w(crop, H=H, Wmax=Wmax)
            arr = np.asarray(patch, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            X.append(arr[None, :, :])              # (1,H,Wmax)
            y.append(word2id[word])
    X = np.stack(X, axis=0) if X else np.zeros((0,1,H,Wmax), np.float32)
    y = np.array(y, dtype=np.int64)
    np.save(out_x, X); np.save(out_y, y)
    print(f"[prepare_numpy] {split_dir} -> {X.shape}, saved: {out_x}, {out_y}")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.vocab, "r", encoding="utf-8") as f:
        word2id = json.load(f)

    # train → train/val로 분리 (0.9/0.1)
    tr_dir = os.path.join(args.img_root, "train")
    te_dir = os.path.join(args.img_root, "test")
    # 일단 한 번에 만들어서 섞고 나눈다
    tmp_x = os.path.join(args.out_dir, "_tmp_train_X.npy")
    tmp_y = os.path.join(args.out_dir, "_tmp_train_y.npy")
    process_split(tr_dir, word2id, tmp_x, tmp_y, H=args.h, Wmax=args.wmax)
    X = np.load(tmp_x); y = np.load(tmp_y)
    if len(y) == 0:
        raise RuntimeError("train에서 usable sample이 없습니다. vocab을 확인하세요.")
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(round(len(y) * args.val_ratio)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    np.save(os.path.join(args.out_dir, "train_X.npy"), X[train_idx])
    np.save(os.path.join(args.out_dir, "train_y.npy"), y[train_idx])
    np.save(os.path.join(args.out_dir, "val_X.npy"),   X[val_idx])
    np.save(os.path.join(args.out_dir, "val_y.npy"),   y[val_idx])
    os.remove(tmp_x); os.remove(tmp_y)

    # test
    process_split(te_dir, word2id,
                  os.path.join(args.out_dir, "test_X.npy"),
                  os.path.join(args.out_dir, "test_y.npy"),
                  H=args.h, Wmax=args.wmax)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True, help="mocr/dataset 루트")
    ap.add_argument("--vocab", required=True, help="artifacts/vocab/word2id.json")
    ap.add_argument("--out_dir", required=True, help="artifacts/npy")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--wmax", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    main(args)
