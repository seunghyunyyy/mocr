import argparse, json, glob, unicodedata, os
from collections import Counter
from tqdm import tqdm

def normalize_token(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    return s

def main(args):
    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    freq = Counter()
    for jp in tqdm(json_paths, desc="scan json (build_vocab)"):
        with open(jp, "r", encoding="utf-8") as f:
            j = json.load(f)
        for b in j.get("bbox", []):
            w = normalize_token(str(b.get("data", "")))
            if w: freq[w] += 1

    total = sum(freq.values())
    items = freq.most_common()
    cum, N = 0, 0
    for i, (_, c) in enumerate(items, 1):
        cum += c
        if cum / total >= args.coverage:
            N = i; break
    if args.max_vocab is not None:
        N = min(N, args.max_vocab)

    os.makedirs(args.out_dir, exist_ok=True)
    top_words = [w for w, _ in items[:N]]
    word2id = {w:i for i, w in enumerate(top_words)}
    id2word = {i:w for w, i in word2id.items()}

    with open(os.path.join(args.out_dir, "word2id.json"), "w", encoding="utf-8") as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "id2word.json"), "w", encoding="utf-8") as f:
        json.dump(id2word, f, ensure_ascii=False, indent=2)

    # 통계 CSV
    with open(os.path.join(args.out_dir, "stats.csv"), "w", encoding="utf-8") as f:
        f.write("word,count,cum_ratio\n")
        cum = 0
        for i, (_, c) in enumerate(tqdm(items, desc="cum coverage"), 1):
            cum += c
            f.write(f"{w},{c},{cum/total:.6f}\n")

    print(f"[build_vocab] total tokens={total}, vocab_size={len(word2id)}, saved to {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="dataset/train JSON 폴더")
    ap.add_argument("--out_dir", required=True, help="저장 폴더 (artifacts/vocab)")
    ap.add_argument("--coverage", type=float, default=0.92, help="누적 커버리지 목표")
    ap.add_argument("--max_vocab", type=int, default=None, help="(선택) 상한")
    args = ap.parse_args()
    main(args)
