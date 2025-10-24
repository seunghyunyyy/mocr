# wordcnn/eval.py
# coding: utf-8
import argparse, json, numpy as onp, sys, os
from tqdm import trange
from common.xp import xp, asnumpy, DTYPE
from wordcnn.model import WordCNN

def load_ckpt_inplace(net, ckpt_path):
    z = onp.load(ckpt_path, allow_pickle=True)
    loaded = 0
    for k in net.params:
        if k in z.files and net.params[k].shape == z[k].shape:
            net.params[k][...] = xp.asarray(z[k])   # in-place 복사(참조 유지)
            loaded += 1
    return z, loaded

def safe_word(id2, i):
    if id2 is None: return str(i)
    if 0 <= i < len(id2):
        w = id2[i]
        return "" if w is None else str(w)
    return "<OOR>"

def logsumexp(a, axis=1, keepdims=True):
    m = a.max(axis=axis, keepdims=True)
    return m + onp.log(onp.exp(a - m).sum(axis=axis, keepdims=True))

def main(args):
    # 1) id2word & ckpt out-dim 체크(있으면)
    id2 = None
    vocab_len = None
    if args.id2word:
        with open(args.id2word, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # id2word가 dict("0":"…")인 경우 리스트로 정규화
        if isinstance(raw, dict):
            tmp = {int(k): v for k, v in raw.items()}
            id2 = [tmp.get(i, None) for i in range(max(tmp.keys())+1)]
        else:
            id2 = raw
        vocab_len = len(id2)

    z_tmp = onp.load(args.ckpt, allow_pickle=True)
    out_dim = int(z_tmp["W4"].shape[1])
    if vocab_len is not None and out_dim != vocab_len:
        print(f"[warn] ckpt out_dim({out_dim}) != len(id2word)({vocab_len})", file=sys.stderr)
    num_classes = out_dim

    # 2) 데이터 로드
    X = onp.load(args.test_x, mmap_mode="r")
    y = onp.load(args.test_y, mmap_mode="r")
    print(f"[data] X={X.shape} DTYPE_infer={DTYPE}  y={y.shape} int?={onp.issubdtype(y.dtype, onp.integer)}")

    # 3) 모델 & 가중치 로드
    net = WordCNN(num_classes=num_classes)
    _, loaded = load_ckpt_inplace(net, args.ckpt)
    print(f"[ckpt] loaded {loaded} params, out_dim={int(net.params['W4'].shape[1])}")

    # 4) prior 준비(선택)
    prior = None
    if args.prior_from:
        ytr = onp.load(args.prior_from, mmap_mode="r")
        cnt = onp.bincount(ytr, minlength=num_classes)
        prior = onp.log(cnt + 1.0)  # 간단 스무딩
        print(f"[prior] from={args.prior_from}  alpha={args.alpha}  nonzero={int((cnt>0).sum())}")

    # 5) (선택) CSV 스트리밍 준비
    writer = None
    if args.dump_csv:
        os.makedirs(os.path.dirname(args.dump_csv) or ".", exist_ok=True)
        # Excel 호환을 위해 utf-8-sig
        fp = open(args.dump_csv, "w", encoding="utf-8-sig")
        fp.write("idx,pred_id,pred_word,pred_prob,gt_id,gt_word,correct")
        if args.topk > 1:
            fp.write(",topk_ids,topk_words,topk_probs")
        fp.write("\n")
        writer = fp

    # 6) 평가 + 예측 출력
    bs = int(args.batch)
    seen = 0
    correct_top1 = 0
    correct_topk = 0
    iters = (len(X) + bs - 1) // bs
    printed = 0

    pbar = trange(iters, desc="eval", unit="batch")
    for it in pbar:
        s = it * bs
        e = min(len(X), s + bs)
        xb = xp.asarray(X[s:e], dtype=DTYPE)
        logits = net.predict(xb)                         # (B, C)
        logits = logits - logits.max(axis=1, keepdims=True)  # 안정화
        logits = asnumpy(logits)

        # prior 보정
        if prior is not None and args.alpha != 0.0:
            logits = logits - float(args.alpha) * prior[None, :]

        ytrue = y[s:e]
        # top-1
        pred = logits.argmax(axis=1)
        correct = (pred == ytrue)
        correct_top1 += int(correct.sum())

        # top-k
        topk_ids = None; topk_probs = None
        if args.topk > 1:
            part = onp.argpartition(-logits, kth=args.topk-1, axis=1)[:, :args.topk]
            # 간단 softmax 확률
            log_den = logsumexp(logits, axis=1, keepdims=True)  # (B,1)
            topk_probs = onp.exp(logits[onp.arange(len(logits))[:, None], part] - log_den)
            hit = (part == ytrue[:, None]).any(axis=1)
            correct_topk += int(hit.sum())
            topk_ids = part

        # 확률(softmax) for top-1
        log_den_full = logsumexp(logits, axis=1, keepdims=True)
        pred_prob = onp.exp(logits[onp.arange(len(pred)), pred] - log_den_full.ravel())

        # (선택) stdout 몇 개 보여주기
        if args.show_n > 0 and printed < args.show_n:
            for i in range(min(args.show_n - printed, len(pred))):
                gi = int(ytrue[i]); pi = int(pred[i])
                gw = safe_word(id2, gi); pw = safe_word(id2, pi)
                ok = "✓" if gi == pi else "✗"
                if args.topk > 1:
                    tk_ids = topk_ids[i].tolist()
                    tk_words = [safe_word(id2, t) for t in tk_ids]
                    tk_ps = topk_probs[i].tolist()
                    print(f"[{s+i}] pred='{pw}'({pi}) p={pred_prob[i]:.3f}  | gt='{gw}'({gi}) {ok}  | top{args.topk}={list(zip(tk_words, tk_ps))}")
                else:
                    print(f"[{s+i}] pred='{pw}'({pi}) p={pred_prob[i]:.3f}  | gt='{gw}'({gi}) {ok}")
                printed += 1
                if printed >= args.show_n: break

        # (선택) CSV 스트리밍 저장
        if writer is not None:
            for i in range(len(pred)):
                gi = int(ytrue[i]); pi = int(pred[i])
                gw = safe_word(id2, gi); pw = safe_word(id2, pi)
                ok = int(gi == pi)
                if args.topk > 1:
                    tk_ids = topk_ids[i].tolist()
                    tk_words = [safe_word(id2, t) for t in tk_ids]
                    tk_ps = topk_probs[i].tolist()
                    writer.write(f"{s+i},{pi},{pw},{pred_prob[i]:.6f},{gi},{gw},{ok},\"{tk_ids}\",\"{tk_words}\",\"{tk_ps}\"\n")
                else:
                    writer.write(f"{s+i},{pi},{pw},{pred_prob[i]:.6f},{gi},{gw},{ok}\n")

        seen += (e - s)
        acc1 = correct_top1 / seen
        if args.topk > 1:
            acck = correct_topk / seen
            pbar.set_postfix(avg_acc=f"{acc1:.4f}", topk=f"{acck:.4f}", seen=seen, total=len(X))
        else:
            pbar.set_postfix(avg_acc=f"{acc1:.4f}", seen=seen, total=len(X))

    if writer is not None:
        writer.close()

    msg = f"\n[EVAL] samples={len(X)}  top1_acc={correct_top1/len(X):.4f}"
    if args.topk > 1:
        msg += f"  top{args.topk}_acc={correct_topk/len(X):.4f}"
    print(msg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_x", required=True)
    ap.add_argument("--test_y", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--id2word", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--prior_from", default=None, help="train_y.npy 경로")
    ap.add_argument("--alpha", type=float, default=1.0, help="prior 보정 세기(0이면 미사용)")
    # ↓↓↓ 추가: 무엇으로 인식했는지 보기용
    ap.add_argument("--show_n", type=int, default=0, help="터미널에 예측/정답 샘플 N개 출력")
    ap.add_argument("--dump_csv", default=None, help="예측 전체를 CSV로 저장(utf-8-sig)")
    args = ap.parse_args()
    main(args)