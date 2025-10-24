# visualize_logs.py
# Usage examples:
#   python visualize_logs.py --csv artifacts/train_log.csv --outdir artifacts/plots
#   python visualize_logs.py --csv artifacts/train_log.csv --outdir artifacts/plots --smooth 3 --ema 0.2 --annotate-best --annotate-minloss
#   python visualize_logs.py --csv artifacts/train_log.csv --yscale-loss log --dpi 200 --figsize 8 5

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_num(s): return pd.to_numeric(s, errors="coerce")

def moving_average(y, k:int|None):
    if not k or k <= 1: return y
    return pd.Series(y).rolling(window=int(k), min_periods=1).mean().to_numpy()

def exponential_ma(y, alpha:float|None):
    if not alpha or alpha <= 0 or alpha >= 1: return y
    out = np.empty_like(y, dtype=float)
    m = None
    for i, v in enumerate(y):
        v = float(v) if v is not None else np.nan
        if i == 0 or np.isnan(v):
            m = v
        else:
            m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out

def plot_xy(x, ys:dict, title, xlabel, ylabel, outpath, yscale=None, annotate_points:dict|None=None, dpi=150, figsize=None):
    if figsize:
        try:
            w, h = map(float, figsize)
            plt.figure(figsize=(w, h))
        except Exception:
            plt.figure()
    else:
        plt.figure()
    for label, y in ys.items():
        plt.plot(x, y, marker="o", label=label)
    if yscale in {"log","symlog"}:
        plt.yscale(yscale)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if len(ys) > 1:
        plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if annotate_points:
        for label, pts in annotate_points.items():
            for (xi, yi, txt) in pts:
                if np.isfinite(yi):
                    plt.scatter([xi],[yi])
                    plt.annotate(txt, (xi, yi), textcoords="offset points", xytext=(6,6))
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to train_log.csv")
    ap.add_argument("--outdir", default="artifacts/plots", help="directory to save figures")
    ap.add_argument("--smooth", type=int, default=0, help="simple moving average window (epochs)")
    ap.add_argument("--ema", type=float, default=0.0, help="exponential moving average alpha (0~1). e.g., 0.2")
    ap.add_argument("--yscale-loss", default="linear", choices=["linear","log","symlog"], help="y-scale for loss")
    ap.add_argument("--yscale-acc", default="linear", choices=["linear","log","symlog"], help="y-scale for accuracy")
    ap.add_argument("--annotate-best", action="store_true", help="annotate best validation accuracy point")
    ap.add_argument("--annotate-minloss", action="store_true", help="annotate minimum loss point")
    ap.add_argument("--dpi", type=int, default=150, help="figure dpi")
    ap.add_argument("--figsize", nargs=2, metavar=("W","H"), help="figure size in inches, e.g., 8 5")
    ap.add_argument("--export-summary", action="store_true", help="also write summary.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    if "epoch" not in df.columns:
        raise ValueError("CSV must contain 'epoch' column.")
    df["epoch"] = to_num(df["epoch"]).fillna(0).astype(int)
    df = df.sort_values("epoch").reset_index(drop=True)

    # -------------------- LOSS --------------------
    if "loss" in df.columns:
        loss = to_num(df["loss"]).to_numpy()
        loss_s = moving_average(loss, args.smooth)
        loss_s = exponential_ma(loss_s, args.ema)
        ann = {}
        if args.annotate_minloss and np.isfinite(loss).any():
            i = int(np.nanargmin(loss))
            ann["loss"] = [(int(df["epoch"][i]), float(loss[i]), f"min={loss[i]:.4f} @ep{int(df['epoch'][i])}")]
        plot_xy(
            df["epoch"],
            {"loss": loss_s},
            title=f"Training Loss per Epoch"
                  + (f" (SMA {args.smooth})" if args.smooth else "")
                  + (f" (EMA α={args.ema})" if args.ema else ""),
            xlabel="Epoch", ylabel="Loss",
            outpath=outdir / "training_loss.png",
            yscale=args.yscale_loss, annotate_points=ann,
            dpi=args.dpi, figsize=args.figsize
        )

    # -------------------- ACCURACY --------------------
    has_train = "train_acc" in df.columns
    has_val   = "val_acc" in df.columns
    if has_train or has_val:
        ys = {}
        ann = {}
        if has_train:
            y = to_num(df["train_acc"]).to_numpy()
            y = moving_average(y, args.smooth)
            y = exponential_ma(y, args.ema)
            ys["train_acc"] = y
        if has_val:
            yv = to_num(df["val_acc"]).to_numpy()
            ys["val_acc"] = exponential_ma(moving_average(yv, args.smooth), args.ema)
            if getattr(args, "annotate_best", False):
                j = int(np.nanargmax(yv))
                ann.setdefault("val_acc", []).append(
                    (int(df["epoch"][j]), float(yv[j]), f"best={yv[j]:.4f} @ep{int(df['epoch'][j])}")
                )
        plot_xy(
            df["epoch"], ys,
            title=f"Accuracy per Epoch"
                  + (f" (SMA {args.smooth})" if args.smooth else "")
                  + (f" (EMA α={args.ema})" if args.ema else ""),
            xlabel="Epoch", ylabel="Accuracy",
            outpath=outdir / "training_accuracy.png",
            yscale=args.yscale_acc, annotate_points=ann if ann else None,
            dpi=args.dpi, figsize=args.figsize
        )

    # -------------------- TIME (per-epoch) --------------------
    if "time_s" in df.columns:
        t = to_num(df["time_s"])
        per_epoch = t.diff().fillna(t).to_numpy()
        y = exponential_ma(moving_average(per_epoch, args.smooth), args.ema)
        plot_xy(
            df["epoch"], {"seconds/epoch": y},
            title=f"Per-Epoch Duration (s)"
                  + (f" (SMA {args.smooth})" if args.smooth else "")
                  + (f" (EMA α={args.ema})" if args.ema else ""),
            xlabel="Epoch", ylabel="Seconds",
            outpath=outdir / "epoch_time.png",
            yscale="linear", annotate_points=None,
            dpi=args.dpi, figsize=args.figsize
        )

    # -------------------- Optional summary --------------------
    if args.export_summary:
        summary = {
            "epochs_logged": int(df["epoch"].max()),
            "min_loss": float(to_num(df["loss"]).min(skipna=True)) if "loss" in df.columns else np.nan,
            "max_train_acc": float(to_num(df["train_acc"]).max(skipna=True)) if "train_acc" in df.columns else np.nan,
            "max_val_acc": float(to_num(df["val_acc"]).max(skipna=True)) if "val_acc" in df.columns else np.nan,
            "avg_epoch_sec": float(pd.Series(per_epoch).mean()) if "time_s" in df.columns else np.nan,
        }
        pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)
        print(f"[summary] wrote {outdir/'summary.csv'}")

    print(f"[done] figures saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
