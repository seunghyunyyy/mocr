# util/prepare_numpy_fast.py
# coding: utf-8
import os, sys, glob, argparse, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from numpy.lib.format import open_memmap as open_npy_memmap  # ← .npy 헤더 포함 memmap

# --- 빠른 JSON 파서 (있으면 orjson)
try:
    import orjson
    def load_json(p):
        with open(p, "rb") as f:
            return orjson.loads(f.read())
except Exception:
    def load_json(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

# --- 이미지 로더 백엔드
def _load_gray_cv2(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H,W), uint8
    if img is None:
        raise RuntimeError(f"imread failed: {path}")
    return img

def _load_gray_pil(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("L"), dtype=np.uint8)  # (H,W)

# --- 라벨 로딩
def load_vocab(vocab_json):
    with open(vocab_json, "r", encoding="utf-8") as f:
        return json.load(f)  # word->id

def _resize_pad_to(img, H, Wmax):
    """img: (H0,W0) uint8 → 목표 (H, Wmax)
       세로 리사이즈 후 오른쪽 0패딩. 비율 유지."""
    h0, w0 = img.shape
    if h0 != H:
        ys = (np.linspace(0, h0 - 1, H)).astype(np.int32)  # 최근접 보간
        img = img[ys][:, :]
        h0, w0 = img.shape
    w = min(Wmax, w0)
    out = np.zeros((H, Wmax), dtype=np.uint8)
    out[:, :w] = img[:, :w]
    return out

def _extract_word(img, meta):
    """bbox를 이용해 단어 영역 crop. meta 스키마에 맞춰 key 수정."""
    bbox = meta.get("bbox") or meta.get("word_bbox") or meta.get("rect")
    if isinstance(bbox, dict):
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    else:
        x, y, w, h = bbox  # [x,y,w,h]
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    return img[y:y+h, x:x+w]

def _one_example(args):
    """워커에서 실행: (index, img_path, json_path, H, Wmax, backend, w2id)
       -> (idx, ok, arr(float32 HxW [-1,1]), label_id)"""
    idx, img_path, json_path, H, Wmax, backend, w2id = args

    # 1) load image
    if backend == "cv2":
        img = _load_gray_cv2(img_path)
    else:
        img = _load_gray_pil(img_path)

    # 2) meta & label (스키마 맞게 키 수정)
    meta = load_json(json_path)
    word = meta.get("text") or meta.get("label") or meta.get("word")
    if not word or (word not in w2id):
        return idx, False, None, None

    # 3) crop & resize+pad
    try:
        roi = _extract_word(img, meta)
    except Exception:
        roi = img
    out = _resize_pad_to(roi, H, Wmax)

    # 4) 정규화 [-1,1]
    arr = out.astype(np.float32) / 127.5 - 1.0

    # 5) label id
    y = int(w2id[word])
    return idx, True, arr, y

def _iter_files(root):
    # root/{train,val,test}/*.(png|jpg) & 같은 이름의 .json 이 있다고 가정
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    for split in ("train", "val", "test"):
        img_files = []
        for ext in exts:
            img_files += glob.glob(os.path.join(root, split, ext))
        img_files.sort()
        pairs = []
        for p in img_files:
            base, _ = os.path.splitext(p)
            jp = base + ".json"
            if os.path.exists(jp):
                pairs.append((p, jp))
        yield split, pairs

def _open_tmp_memmap(out_dir, split, N, H, Wmax, dtype=np.float32):
    os.makedirs(out_dir, exist_ok=True)
    final_X = os.path.join(out_dir, f"{split}_X.npy")
    final_Y = os.path.join(out_dir, f"{split}_y.npy")
    tmp_X   = final_X + ".tmp"
    tmp_Y   = final_Y + ".tmp"
    # .npy 포맷으로 바로 memmap
    Xmm = open_npy_memmap(tmp_X, mode="w+", dtype=dtype, shape=(N, 1, H, Wmax))
    Ymm = open_npy_memmap(tmp_Y, mode="w+", dtype=np.int64, shape=(N,))
    return Xmm, Ymm, final_X, final_Y, tmp_X, tmp_Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True, help="dataset 루트 (train/val/test 하위)")
    ap.add_argument("--vocab",    required=True, help="word2id.json")
    ap.add_argument("--out_dir",  required=True, help="출력 npy 디렉토리")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--wmax", type=int, default=256)
    ap.add_argument("--backend", choices=["cv2", "pil"], default="cv2")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--chunksize", type=int, default=512)  # 현재 구현에선 참고용
    args = ap.parse_args()

    w2id = load_vocab(args.vocab)

    for split, pairs in _iter_files(args.img_root):
        if not pairs:
            print(f"[{split}] no files"); continue

        H, Wmax = args.h, args.wmax
        N = len(pairs)
        Xmm, Ymm, Xpath, Ypath, Xtmp, Ytmp = _open_tmp_memmap(args.out_dir, split, N, H, Wmax)
        print(f"[{split}] files={N}  → {Xpath}, {Ypath}")

        # 작업 튜플
        jobs = [(idx, ip, jp, H, Wmax, args.backend, w2id)
                for idx, (ip, jp) in enumerate(pairs)]

        success_mask = np.zeros(N, dtype=bool)
        ok = 0

        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex, \
             tqdm(total=N, desc=f"{split}", unit="img") as pbar:

            futures = [ex.submit(_one_example, j) for j in jobs]
            for fut in as_completed(futures):
                idx, success, arr, y = fut.result()
                if success:
                    Xmm[idx, 0, :, :] = arr
                    Ymm[idx] = y
                    success_mask[idx] = True
                    ok += 1
                pbar.update(1)
                if ok and (ok % 5000 == 0):
                    pbar.set_postfix(ok=ok, kept_ratio=f"{ok/N:.2%}")

        # flush & close memmaps
        Xmm.flush(); Ymm.flush()
        del Xmm; del Ymm

        # 최종화
        if success_mask.all():
            # 모든 샘플 성공: 임시 → 최종 경로로 원자적 이동
            os.replace(Xtmp, Xpath)
            os.replace(Ytmp, Ypath)
            print(f"[{split}] done. kept {ok}/{N}")
        else:
            # 일부 실패: 성공 인덱스만 압축 저장
            Xsrc = open_npy_memmap(Xtmp, mode="r")
            Ysrc = open_npy_memmap(Ytmp, mode="r")
            sel = np.where(success_mask)[0]
            np.save(Xpath, np.asarray(Xsrc[sel]))
            np.save(Ypath, np.asarray(Ysrc[sel]))
            del Xsrc; del Ysrc
            os.remove(Xtmp); os.remove(Ytmp)
            print(f"[{split}] kept {len(sel)}/{N}")

if __name__ == "__main__":
    main()