# util/visualize_all.py
# coding: utf-8
import os, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_train_curves(log_df, outdir: Path):
    made = []

    # Loss vs Epoch
    if "loss" in log_df.columns:
        fig, ax = plt.subplots(figsize=(7,4.2), dpi=120)
        ax.plot(log_df["epoch"], log_df["loss"], label="loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training Loss per Epoch")
        ax.legend(); fig.tight_layout()
        p = outdir / "train_loss_epoch.png"; fig.savefig(p); plt.close(fig)
        made.append(p)

    # Accuracy vs Epoch
    if ("train_acc" in log_df.columns) or ("val_acc" in log_df.columns):
        fig, ax = plt.subplots(figsize=(7,4.2), dpi=120)
        if "train_acc" in log_df.columns:
            ax.plot(log_df["epoch"], log_df["train_acc"], label="train_acc")
        if "val_acc" in log_df.columns:
            ax.plot(log_df["epoch"], log_df["val_acc"], label="val_acc")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_title("Accuracy per Epoch")
        ax.legend(); fig.tight_layout()
        p = outdir / "acc_epoch.png"; fig.savefig(p); plt.close(fig)
        made.append(p)

    return made


def reliability_table_and_plot(preds, outdir: Path):
    if "pred_prob" not in preds.columns or preds["pred_prob"].isna().all():
        return []

    df = preds.dropna(subset=["pred_prob"]).copy()
    df["bin"] = np.minimum((df["pred_prob"]*10).astype(int), 9)  # 0~9
    cal = df.groupby("bin").agg(conf=("pred_prob","mean"),
                                acc=("correct","mean"),
                                cnt=("correct","size")).reset_index()

    out = []
    fig, ax = plt.subplots(figsize=(5,5), dpi=120)
    xs = np.linspace(0,1,100)
    ax.plot(xs, xs, label="ideal")
    ax.plot(cal["conf"], cal["acc"], label="model")
    ax.set_title("Reliability Diagram"); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.legend(); fig.tight_layout()
    p = outdir / "reliability.png"; fig.savefig(p); plt.close(fig)
    out.append(p)

    cal_path = outdir / "reliability_table.csv"
    cal.to_csv(cal_path, index=False, encoding="utf-8-sig")
    out.append(cal_path)
    return out


def plot_overall_and_running(preds, outdir: Path):
    out = []

    # Overall accuracy bar
    n = len(preds)
    acc = float(preds["correct"].mean()) if n else float("nan")
    fig, ax = plt.subplots(figsize=(4,4), dpi=120)
    ax.bar(["accuracy"], [acc])
    ax.set_ylim(0,1); ax.set_ylabel("top-1 accuracy")
    ax.set_title(f"Overall accuracy = {acc:.4f} (n={n})")
    fig.tight_layout(); p = outdir/"overall_accuracy.png"; fig.savefig(p); plt.close(fig)
    out.append(p)

    # Correct vs Wrong counts
    counts = preds["correct"].value_counts().reindex([1,0], fill_value=0)
    fig, ax = plt.subplots(figsize=(5,4), dpi=120)
    ax.bar(["correct","wrong"], [int(counts.get(1,0)), int(counts.get(0,0))])
    ax.set_title("Counts: correct vs wrong"); ax.set_ylabel("count")
    fig.tight_layout(); p = outdir/"correct_wrong_counts.png"; fig.savefig(p); plt.close(fig)
    out.append(p)

    # Running accuracy curve (샘플 순서대로 누적 정확도)
    run = preds["correct"].expanding().mean()
    fig, ax = plt.subplots(figsize=(7,4.2), dpi=120)
    ax.plot(run.values)
    ax.set_title("Running Accuracy over Samples"); ax.set_xlabel("sample index"); ax.set_ylabel("accuracy")
    fig.tight_layout(); p = outdir/"running_accuracy.png"; fig.savefig(p); plt.close(fig)
    out.append(p)

    return out


def plot_acc_by_gt_len(preds, outdir: Path):
    out = []
    lens = preds["gt_word"].fillna("").astype(str).map(len)
    g = pd.DataFrame({"len": lens, "correct": preds["correct"]}).groupby("len", as_index=False)["correct"].mean()
    fig, ax = plt.subplots(figsize=(7,4.2), dpi=120)
    ax.plot(g["len"], g["correct"])
    ax.set_title("Accuracy vs. GT Word Length")
    ax.set_xlabel("length"); ax.set_ylabel("accuracy")
    fig.tight_layout(); p = outdir/"acc_vs_len.png"; fig.savefig(p); plt.close(fig)
    out.append(p)
    return out


def plot_acc_by_freq_bins(preds, outdir: Path, q=10):
    """GT 단어 빈도 기준으로 q-분위(기본 10분위) 별 평균 정확도."""
    out = []
    gt = preds["gt_word"].fillna("").astype(str)
    freq = gt.value_counts()
    preds = preds.copy()
    preds["gt_freq"] = gt.map(freq)

    try:
        preds["freq_bin"] = pd.qcut(preds["gt_freq"], q=q, duplicates="drop")
        gb = preds.groupby("freq_bin").agg(acc=("correct","mean"),
                                           cnt=("correct","size"),
                                           fmin=("gt_freq","min"),
                                           fmax=("gt_freq","max")).reset_index()
        fig, ax = plt.subplots(figsize=(7,4.2), dpi=120)
        ax.plot(range(len(gb)), gb["acc"])
        ax.set_title(f"Accuracy by GT Frequency (q={len(gb)})")
        ax.set_xlabel("frequency bin (low→high)"); ax.set_ylabel("accuracy")
        fig.tight_layout(); p = outdir/"acc_by_freq_bins.png"; fig.savefig(p); plt.close(fig)
        out.append(p)
        gb.to_csv(outdir/"acc_by_freq_bins.csv", index=False, encoding="utf-8-sig"); out.append(outdir/"acc_by_freq_bins.csv")
    except Exception as e:
        # 데이터가 적거나 전부 같은 빈도일 때 qcut이 실패할 수 있음
        pass

    return out


def plot_top_classes_accuracy(preds, outdir: Path, topn=20):
    """GT 기준 상위 topn 빈도 클래스의 정확도 막대 그래프."""
    out = []
    gt = preds["gt_word"].fillna("").astype(str)
    freq = gt.value_counts()
    top = freq.head(topn).index
    sub = preds[gt.isin(top)].copy()
    by_cls = sub.groupby("gt_word").agg(n=("correct","size"), acc=("correct","mean")).reset_index()
    by_cls = by_cls.sort_values("n", ascending=False)

    fig, ax = plt.subplots(figsize=(10,5), dpi=120)
    ax.bar(by_cls["gt_word"].astype(str), by_cls["acc"])
    ax.set_ylim(0,1); ax.set_title(f"Accuracy of Top-{topn} Frequent GT Words")
    ax.set_ylabel("accuracy"); ax.set_xlabel("gt_word")
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    fig.tight_layout(); p = outdir/"acc_top_freq_classes.png"; fig.savefig(p); plt.close(fig)
    out.append(p)

    by_cls.to_csv(outdir/"acc_top_freq_classes.csv", index=False, encoding="utf-8-sig")
    out.append(outdir/"acc_top_freq_classes.csv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_log", default="train_log.csv", help="학습 로그 CSV 경로")
    ap.add_argument("--pred_csv",  default="test_pred.csv",  help="추론 결과 CSV 경로 (idx,pred_id,pred_word,pred_prob,gt_id,gt_word,correct,...)")
    ap.add_argument("--outdir",    default="artifacts/plots", help="출력 폴더")
    ap.add_argument("--head", type=int, default=50, help="미리보기 저장 행 수")
    ap.add_argument("--topn", type=int, default=20, help="상위 N개 빈도 클래스 정확도 막대 그래프")
    ap.add_argument("--q",    type=int, default=10, help="빈도 분위수 개수(qcut)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / ts
    outdir.mkdir(parents=True, exist_ok=True)

    made_files = []

    # 1) Training log
    tl = Path(args.train_log)
    if tl.exists():
        log_df = pd.read_csv(tl, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        ensure_float(log_df, ["epoch","iter_global","iter_in_epoch","loss","train_acc","val_acc","time_s"])
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        made_files += plot_train_curves(log_df, outdir)
        log_df.to_csv(outdir/"train_log_preview.csv", index=False, encoding="utf-8-sig"); made_files.append(outdir/"train_log_preview.csv")
    else:
        print(f"[warn] train_log not found: {tl}")

    # 2) Predictions
    pc = Path(args.pred_csv)
    if pc.exists():
        preds = pd.read_csv(pc, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        preds.columns = [c.strip() for c in preds.columns]

        if "correct" in preds.columns:
            preds["correct"] = pd.to_numeric(preds["correct"], errors="coerce").fillna(0).astype(int)
        else:
            preds["correct"] = 0

        if "pred_prob" in preds.columns:
            preds["pred_prob"] = pd.to_numeric(preds["pred_prob"], errors="coerce")
        else:
            preds["pred_prob"] = np.nan

        for c in ["pred_word","gt_word"]:
            if c not in preds.columns: preds[c] = ""

        # 요약 파일
        n = len(preds); acc = float(preds["correct"].mean()) if n else float("nan")
        with open(outdir/"summary.txt","w",encoding="utf-8") as f:
            f.write(f"rows={n}\n")
            f.write(f"top1_acc={acc:.4f}\n")
        made_files.append(outdir/"summary.txt")

        # (NEW) Accuracy 시각화 묶음
        made_files += plot_overall_and_running(preds, outdir)
        made_files += plot_acc_by_gt_len(preds, outdir)
        made_files += plot_acc_by_freq_bins(preds, outdir, q=max(2, args.q))
        made_files += plot_top_classes_accuracy(preds, outdir, topn=max(1, args.topn))

        # Confidence 기반(있을 때만)
        made_files += reliability_table_and_plot(preds, outdir)

        # 미리보기 저장
        preds.head(args.head).to_csv(outdir/"pred_head.csv", index=False, encoding="utf-8-sig")
        made_files.append(outdir/"pred_head.csv")
    else:
        print(f"[warn] pred_csv not found: {pc}")

    with open(outdir/"README_outputs.txt","w",encoding="utf-8") as f:
        f.write("Generated plots/tables:\n")
        for p in made_files:
            f.write(str(Path(p).name)+"\n")

    print("\n[✓] Done. Files written under:", outdir)


if __name__ == "__main__":
    main()
