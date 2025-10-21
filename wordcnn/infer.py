import argparse, os, glob, json, numpy as np
from PIL import Image
from wordcnn.model import WordCNN
from tqdm import tqdm

def crop_from_quad(img, xs, ys, pad=2):
    x1, x2 = max(0, min(xs)-pad), min(img.width, max(xs)+pad)
    y1, y2 = max(0, min(ys)-pad), min(img.height, max(ys)+pad)
    return img.crop((x1, y1, x2, y2))

def resize_h_pad_w(img, H=32, Wmax=256):
    w, h = img.size
    new_w = max(1, int(round(w * (H / h))))
    img = img.resize((new_w, H), Image.BILINEAR)
    if new_w > Wmax:
        img = img.resize((Wmax, H), Image.BILINEAR)
        new_w = Wmax
    canvas = Image.new("L", (Wmax, H), color=0)
    canvas.paste(img, (0,0))
    return canvas

def softmax(a):
    a = a - a.max(axis=1, keepdims=True)
    exp = np.exp(a)
    return exp / exp.sum(axis=1, keepdims=True)

def main(args):
    with open(args.id2word, "r", encoding="utf-8") as f:
        id2word = {int(k):v for k,v in json.load(f).items()}
    num_classes = len(id2word)
    net = WordCNN(num_classes=num_classes)
    z = np.load(args.ckpt); net.params.update({k:z[k] for k in z.files})

    out_lines = ["file,x1,y1,w,h,pred,prob"]
    pngs = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    for ip in tqdm(pngs, desc="infer png"):
        jp = ip[:-4] + ".json"
        if not os.path.exists(jp): continue
        img = Image.open(ip).convert("L")
        with open(jp, "r", encoding="utf-8") as f:
            j = json.load(f)
        bboxes = j.get("bbox", [])
        for b in bboxes:
            xs, ys = b.get("x", []), b.get("y", [])
            if len(xs) != 4 or len(ys) != 4:
                continue
            crop = crop_from_quad(img, xs, ys, pad=2)
            patch = resize_h_pad_w(crop, H=args.h, Wmax=args.wmax)
            x = np.asarray(patch, dtype=np.float32)/255.0
            x = (x-0.5)/0.5
            x = x[None, None, :, :]  # (1,1,H,W)
            logits = net.predict(x)
            from common.xp import xp as np
            prob = asnumpy(softmax(logits))[0]
            pred_id = int(np.argmax(prob))
            pmax = float(prob[pred_id])
            pred = id2word.get(pred_id, "<UNK>")
            if args.reject_tau is not None and pmax < args.reject_tau:
                pred = "UNKNOWN"
            x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
            out_lines.append(f"{os.path.basename(ip)},{x1},{y1},{x2-x1},{y2-y1},\"{pred}\",{pmax:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("[infer] saved:", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--id2word", required=True)
    ap.add_argument("--out", default="artifacts/preds_test.csv")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--wmax", type=int, default=256)
    ap.add_argument("--reject_tau", type=float, default=0.5)
    args = ap.parse_args()
    main(args)
