# src/train_hybrid.py
"""
Обучение Hybrid-модели (DistilBERT CLS + hand_feats + sentiment + num_ent)
с Focal-Loss вместо CrossEntropy.

Запуск:
    python src/train_hybrid.py --epochs 5 --batch_size 16 --lr 2e-5 --device auto
"""

import os
import random
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import (
    pipeline,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)

import spacy
import joblib

# --- project imports -------------------------------------------------
import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")          # путь к репо в Colab

from src.config import (
    TRAIN_PATH, VAL_PATH, MODEL_DIR, OUTPUT_DIR,
    BATCH_SIZE as CFG_BATCH, EPOCHS as CFG_EPOCHS,
    LEARNING_RATE as CFG_LR, SEED,
)
from src.features import HandCraftedFeaturizer
from src.dataset import NewsDataset
from src.models.hybrid import HybridModel
from src.utils.losses import FocalLoss

# ---------------------------------------------------------------------

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser("Train HybridModel")
    p.add_argument("--epochs", "-e", type=int, default=None)
    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    p.add_argument("--head", choices=["linear", "mlp"], # MLP модель
                    default="linear",
                    help="Какая голова: linear (старая) или mlp (новая)")
    return p.parse_args()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # hyper-params
    epochs     = args.epochs     or CFG_EPOCHS
    batch_size = args.batch_size or CFG_BATCH
    lr         = args.lr         or CFG_LR
    
    model_name = "hybrid_mlp" if args.head == "mlp" else "hybrid"


    # -------- load CSVs ------------------------------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    # -------- 1) hand-crafted features --------------------------------
    logger.info("Computing hand-crafted features …")
    featurizer = HandCraftedFeaturizer()
    X_train = featurizer.transform(train_df.text)
    X_val   = featurizer.transform(val_df.text)

    scaler = StandardScaler().fit(X_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    train_df["hand_feats"] = scaler.transform(X_train).tolist()
    val_df["hand_feats"]   = scaler.transform(X_val).tolist()

    # -------- 2) sentiment (HF pipeline) ------------------------------
    device_id = device.index if device.type == "cuda" else -1
    logger.info(f"Creating sentiment pipeline on device {device_id} …")
    sent_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device_id,
        batch_size=32,
    )
    logger.info("Computing sentiment …")
    train_df["sentiment"] = [
        1.0 if x["label"] == "NEGATIVE" else 0.0
        for x in sent_pipe(list(train_df.text))
    ]
    val_df["sentiment"] = [
        1.0 if x["label"] == "NEGATIVE" else 0.0
        for x in sent_pipe(list(val_df.text))
    ]

    # -------- 3) named-entity counts ----------------------------------
    logger.info("Computing named-entity counts …")
    nlp = spacy.load("en_core_web_sm")
    train_df["num_ent"] = [len(nlp(t[:300]).ents) for t in train_df.text]
    val_df["num_ent"]   = [len(nlp(t[:300]).ents) for t in val_df.text]

    # (optionally) persist augmented CSVs
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH,   index=False)
    logger.info("Hybrid features added and CSVs updated.")

    # -------- DataLoaders ---------------------------------------------
    train_ds = NewsDataset(train_df, use_hand_feats=True)
    val_ds   = NewsDataset(val_df,   use_hand_feats=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=val_ds.collate_fn, num_workers=2
    )

    # -------- Model / Optim / Sched -----------------------------------
    # model = HybridModel(num_labels=2).to(device)

    from src.models import get_model
    model = get_model(model_name).to(device)
    logger.info(f"Model '{model_name}' params: {sum(p.numel() for p in model.parameters()):,}")

    logger.info(f"HybridModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = AdamW(model.parameters(), lr=lr)
    tot_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * tot_steps), num_training_steps=tot_steps
    )

    loss_fn = FocalLoss(alpha=0.75, gamma=2.0, reduction="mean")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # best_f1, save_path = 0.0, OUTPUT_DIR / "hybrid_best.pt"
    best_f1 = 0.0
    train_loss_hist,  val_loss_hist  = [], []
    train_f1_hist,    val_f1_hist    = [], []
    save_path = OUTPUT_DIR / f"hybrid_best.pt"



    # =================== TRAIN / VAL LOOP =============================
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            hand  = batch["hand_feats"].to(device)
            sent  = batch["sentiment"].to(device)      # [B,1]
            ne    = batch["num_ent"].to(device)        # [B,1]
            labels= batch["labels"].to(device)

            logits = model(ids, mask, hand_feats=hand, sentiment=sent, num_ent=ne)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        epoch_train_loss = running / len(train_loader)
        train_loss_hist.append(epoch_train_loss)
        logger.info(f"Epoch {ep}/{epochs}  train loss: {running/len(train_loader):.4f}")

        # -------- Validation ------------------------------------------
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            val_running = 0.0
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                hand = batch["hand_feats"].to(device)
                sent = batch["sentiment"].to(device)
                ne   = batch["num_ent"].to(device)
                labels = batch["labels"].to(device)

                logits = model(ids, mask, hand_feats=hand, sentiment=sent, num_ent=ne)
                
                # Записываем логи для графика
                val_loss = loss_fn(logits, labels)
                val_running += val_loss.item()

                preds  = torch.argmax(logits, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        rep = classification_report(
            y_true, y_pred, target_names=["real", "fake"],
            digits=4, output_dict=True
        )
        f1_fake = rep["fake"]["f1-score"]
        acc = (np.array(y_pred) == np.array(y_true)).mean()
        logger.info(f"Val acc={acc:.4f}, fake-F1={f1_fake:.4f}")
        
        val_loss_hist.append(val_running / len(val_loader))
        val_f1_hist.append(f1_fake)

        if f1_fake > best_f1:
            best_f1 = f1_fake
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best hybrid model → {save_path.name}")
        
  

    # -------------------Drawing a graph---------------------------
    # ────────────────────── SAVE CURVES ─────────────────────────
    model_out_dir = OUTPUT_DIR / "hybrid"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) loss_curve.png
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(train_loss_hist, label="train loss", marker="o")
    ax.plot(val_loss_hist,   label="val loss",   marker="s")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout()
    fig.savefig(model_out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # 2) f1_curve.png
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(val_f1_hist,   label="val fake-F1",   marker="s")
    ax.set_xlabel("epoch"); ax.set_ylabel("F1"); ax.legend()
    fig.tight_layout()
    fig.savefig(model_out_dir / "f1_curve.png", dpi=150)
    plt.close(fig)

    logger.info("Saved curves → %s", model_out_dir)
