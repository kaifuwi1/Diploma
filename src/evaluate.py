# ── src/evaluate.py ───────────────────────────────────────────────────
"""
Считает метрики для checkpoint-а модели, строит confusion-matrix
и (опц.) PR-кривую.

Пример:
python src/evaluate.py \
       --model hybrid \
       --ckpt src/outputs/hybrid_best.pt \
       --csv  data/val.csv \
       --save_pr
"""
from __future__ import annotations
import argparse, json, sys, csv, pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    PrecisionRecallDisplay,
)

# ── project imports ──────────────────────────────────────────────────
ROOT = pathlib.Path("/content/drive/MyDrive/FakeNewNet")
sys.path.append(str(ROOT))

from src.config import DEVICE, OUTPUT_DIR
from src.dataset import NewsDataset
from src.models  import get_model
# --------------------------------------------------------------------

@torch.inference_mode()
def _predict(model, loader, use_hand: bool):
    y_true, y_pred, y_prob = [], [], []

    model.eval()
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)

        if use_hand:                              # HybridModel
            logits = model(
                ids, mask,
                hand_feats=batch["hand_feats"].to(DEVICE),
                sentiment=batch["sentiment"].to(DEVICE),
                num_ent=batch["num_ent"].to(DEVICE),
            )
        else:                                     # baseline / bigru / cnn
            logits = model(ids, mask)

        prob = torch.softmax(logits, 1)[:, 1]     # P(fake)
        pred = torch.argmax(logits, 1)

        y_true.extend(batch["labels"].cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def main(csv_path: Path, ckpt_path: Path,
         model_name: str, save_pr: bool):

    df = pd.read_csv(csv_path)
    use_hand = model_name == "hybrid"

    dataset = NewsDataset(df, use_hand_feats=use_hand)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=dataset.collate_fn, num_workers=0
    )

    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    y_true, y_pred, y_prob = _predict(model, loader, use_hand)

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    macro_f1 = f1.mean()
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = None
    pr_auc = average_precision_score(y_true, y_prob)

    metrics = {
        "accuracy":       round(float(acc), 4),
        "macro_F1":       round(float(macro_f1), 4),
        "precision_real": round(float(prec[0]), 4),
        "recall_real":    round(float(rec[0]), 4),
        "f1_real":        round(float(f1[0]), 4),
        "precision_fake": round(float(prec[1]), 4),
        "recall_fake":    round(float(rec[1]), 4),
        "f1_fake":        round(float(f1[1]), 4),
        "roc_auc":        None if roc is None else round(float(roc), 4),
        "pr_auc":         round(float(pr_auc), 4),
    }
    print(json.dumps(metrics))                    # машинный вывод

    # ── Confusion-matrix PNG ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(
        cm, display_labels=["real", "fake"]
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{model_name} – Confusion matrix")
    fig.tight_layout()

    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_png = out_dir / f"cm_{model_name}.png"
    fig.savefig(cm_png, dpi=120)
    plt.close(fig)
    print("saved", cm_png)

    # ── PR-curve PNG (optional) ───────────────────────────────────────
    if save_pr:
        disp = PrecisionRecallDisplay.from_predictions(
            y_true, y_prob, name=model_name, pos_label=1
        )
        pr_fig = disp.figure_
        pr_fig.set_size_inches(4, 4)
        pr_fig.tight_layout()
        pr_png = out_dir / f"pr_{model_name}.png"
        pr_fig.savefig(pr_png, dpi=120)
        plt.close(pr_fig)
        print("saved", pr_png)


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  type=Path, default=ROOT / "data/val.csv",
                    help="CSV with columns text,label,…")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="path to model checkpoint *.pt")
    ap.add_argument("--model", required=True,
                    choices=["baseline", "bigru", "cnn", "hybrid"])
    ap.add_argument("--save_pr", action="store_true",
                    help="save Precision-Recall curve PNG")
    # parse_known_args  → игнорирует Jupyter аргумент -f
    args, _ = ap.parse_known_args()
    main(args.csv, args.ckpt, args.model, args.save_pr)
