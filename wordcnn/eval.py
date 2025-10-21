import argparse, os, json, numpy as np
from wordcnn.model import WordCNN
from tqdm import tqdm

def load_params(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}

def main(args):
    X = np.load(args.test_x); y = np.load(args.test_y)
    with open(args.id2word, "r", encoding="utf-8") as f:
        id2word = {int(k):v for k,v in json.load(f).items()}
    num_classes = len(id2word)
    net = WordCNN(num_classes=num_classes)
    net.params.update(load_params(args.ckpt))

    # Top-1 / Top-5
    bs = 1024
    pred_top1 = 0; pred_top5 = 0; tot = 0
    for i in tqdm(range(0, len(y), bs), desc="eval"):
        xb = X[i:i+bs]; yb = y[i:i+bs]
        pred_top1 += (net.accuracy(xb, yb, topk=1) * len(yb))
        pred_top5 += (net.accuracy(xb, yb, topk=5) * len(yb))
        tot += len(yb)
    acc1 = pred_top1 / tot
    acc5 = pred_top5 / tot
    print(f"[eval] Top-1={acc1:.4f}  Top-5={acc5:.4f}")

    # 길이 구간/문자유형 분석(간단)
    lengths = np.array([len(id2word[int(t)]) for t in y])
    for rng in [(1,3),(4,6),(7,99)]:
        m = (lengths>=rng[0]) & (lengths<=rng[1])
        if m.sum()>0:
            print(f"  len {rng}: {net.accuracy(X[m], y[m], 1):.4f} ({m.sum()} samples)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_x", required=True)
    ap.add_argument("--test_y", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--id2word", required=True)
    args = ap.parse_args()
    main(args)
