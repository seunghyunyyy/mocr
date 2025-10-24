# mocr — Mini OCR (word-level) 🔤

간단한 CNN 기반 단어 분류기(mini OCR) 프로젝트입니다.
`dataset/`에 있는 지필 설문지 스캔본의 **단어 영역**을 고정 크기 이미지로 전처리 → **WordCNN** 학습 → **정확도 평가/시각화** 순서로 진행합니다.


---

## 0) 빠른 실행 요약

아래 순서대로 한 줄씩 실행하세요. (각 단계의 자세한 설명은 아래에 있습니다.)

### 1) 어휘 사전 만들기 (build_vocab)

```bash
python -m util.build_vocab \
  --json_dir dataset/train \
  --out_dir artifacts/vocab \
  --coverage 0.92 --max_vocab 5000
```

### 2) 넘파이 데이터 만들기 (prepare_numpy)

```bash
python -m util.prepare_numpy \
  --img_root dataset \
  --vocab artifacts/vocab/word2id.json \
  --out_dir artifacts/npy \
  --h 32 --wmax 256 --val_ratio 0.1
```

### 3) 학습 (train)

```bash
MOCR_DTYPE=float32 USE_CUPY=1 python -m wordcnn.train \
  --train_x artifacts/npy/train_X.npy \
  --train_y artifacts/npy/train_y.npy \
  --val_x   artifacts/npy/val_X.npy \
  --val_y   artifacts/npy/val_y.npy \
  --epochs 7 --batch 64 --lr 5e-4 \
  --ckpt_dir artifacts/ckpt_fp32 \
  --resume artifacts/ckpt/wordcnn_best.npz \
  --eval_samples 4096 --eval_every 1 --eval_bs 32 \
  --save_every_iters 2000 --save_every_secs 300
```

### 4) 평가 (eval)

```bash
USE_CUPY=1 python -m wordcnn.eval \
  --test_x artifacts/npy/test_X.npy \
  --test_y artifacts/npy/test_y.npy \
  --ckpt artifacts/ckpt/wordcnn_best.npz \
  --id2word artifacts/vocab/id2word.json \
  --batch 64 --topk 5 \
  --prior_from artifacts/npy/train_y.npy --alpha 0.3 \
  --show_n 20 \
  --dump_csv artifacts/infer/test_pred.csv
```

---

## 1) 환경 구성

* Python 3.11~3.12 권장
* conda environment.yaml 설치
* CUDA 12.x + NVIDIA GPU (선택, CuPy 가속)


환경 변수:

* `USE_CUPY=1` → CuPy(GPU) 백엔드 사용, 미설정 시 NumPy(CPU).
* `MOCR_DTYPE=float32|float16` → 학습/추론 내부 부동소수 dtype.

---

## 2) 데이터 구조

```
dataset/
 ├─ train/
 │   ├─ IMG_....png
 │   ├─ IMG_....json
 │   └─ ...
 ├─ val/   (선택. 없으면 prepare_numpy에서 val_ratio로 분리)
 └─ test/
     ├─ IMG_....png
     ├─ IMG_....json
     └─ ...
```

* 각 `.json`은 페이지의 단어 bbox와 텍스트를 포함합니다.
* `util.build_vocab`는 `dataset/train` JSON들에서 **단어 어휘**를 빌드합니다.
* `util.prepare_numpy`는 (bbox로 잘라낸) 단어 이미지를 `(1, H, Wmax)` 모양으로 정규화하여
  `train_X.npy / train_y.npy / val_X.npy / val_y.npy / test_X.npy / test_y.npy`를 생성합니다.

---

## 3) 단계별 설명

### (1) 어휘 사전 빌드

```bash
python -m util.build_vocab \
  --json_dir dataset/train \
  --out_dir artifacts/vocab \
  --coverage 0.92 --max_vocab 5000
```

* `coverage`: 빈도 상위 토큰들로 누적 커버리지를 맞춥니다.
* `max_vocab`: 최대 어휘 수 제한 (단어 분류기 규모/속도에 직접 영향).

출력:

* `artifacts/vocab/word2id.json`
* `artifacts/vocab/id2word.json`

### (2) Numpy 데이터 생성

```bash
python -m util.prepare_numpy \
  --img_root dataset \
  --vocab artifacts/vocab/word2id.json \
  --out_dir artifacts/npy \
  --h 32 --wmax 256 --val_ratio 0.1
```

* H=32로 세로 리사이즈, 가로는 Wmax=256까지 우측 패딩.
* `val_ratio`로 train 내부에서 검증세트를 자동 분리.

출력:

* `artifacts/npy/{train,val,test}_{X,y}.npy`

### (3) 학습

```bash
MOCR_DTYPE=float32 USE_CUPY=1 python -m wordcnn.train \
  --train_x artifacts2/npy/train_X.npy \
  --train_y artifacts2/npy/train_y.npy \
  --val_x   artifacts2/npy/val_X.npy \
  --val_y   artifacts2/npy/val_y.npy \
  --epochs 7 --batch 64 --lr 5e-4 \
  --ckpt_dir artifacts2/ckpt_fp32 \
  --resume artifacts2/ckpt/wordcnn_best.npz \
  --eval_samples 4096 --eval_every 1 --eval_bs 32 \
  --save_every_iters 2000 --save_every_secs 300
```

* 체크포인트는 `*_last.npz`, `*_best.npz`로 저장됩니다.
* `--resume`로 이어서 학습 가능(옵티마이저 상태까지 복구 지원).
* 진행중 중단(CTRL+C) 시 최근 저장 주기 기준으로 체크포인트가 남습니다.

로그/시각화:

* `artifacts/train_log.csv` 기록
* 시각화: `python util/visualize_logs.py --csv artifacts/train_log.csv --outdir artifacts/plots --smooth 3 --ema 0.2`

### (4) 평가

```bash
USE_CUPY=1 python -m wordcnn.eval \
  --test_x artifacts/npy/test_X.npy \
  --test_y artifacts/npy/test_y.npy \
  --ckpt artifacts/ckpt/wordcnn_best.npz \
  --id2word artifacts/vocab/id2word.json \
  --batch 64 --topk 5 \
  --prior_from artifacts/npy/train_y.npy --alpha 0.3 \
  --show_n 20 \
  --dump_csv artifacts/infer/test_pred.csv
```

* `--topk`로 Top-K 정확도 측정.
* `--prior_from + --alpha`로 **클래스 사전확률 보정**(불균형 완화).
* `--dump_csv`로 전체 예측(안전한 CSV quoting 적용)을 파일로 저장.

출력:

* 콘솔: Top-1 / Top-K 정확도
* 파일: `artifacts/infer/test_pred.csv` (idx, pred_id, pred_word, pred_prob, gt_id, gt_word, correct, …)

