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
from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    pipeline
)

import spacy

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import (
    VAL_PATH,
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED,
    OUTPUT_DIR
)
TRAIN_PATH = Path("/content/drive/MyDrive/FakeNewNet/data/train1.csv")

from src.dataset import NewsDataset
from src.models import get_model

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(df: pd.DataFrame):
    counts = df.label.value_counts().to_dict()
    n_real = counts.get("real", 0)
    n_fake = counts.get("fake", 0)
    total = n_real + n_fake
    w_real = total / (n_real + 1e-12)
    w_fake = total / (n_fake + 1e-12)
    mean = (w_real + w_fake) / 2
    return torch.tensor([w_real/mean, w_fake/mean], device=DEVICE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["baseline", "bigru", "cnn", "hybrid"],
                   required=True)
    p.add_argument("--epochs", "-e", type=int, default=None)
    p.add_argument("--batch_size", "-b", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    # ============ DEVICE ============
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ============ PARAMS ============
    epochs     = args.epochs     or EPOCHS
    batch_size = args.batch_size or BATCH_SIZE
    lr         = args.lr         or LEARNING_RATE
    use_hand   = (args.model == "hybrid")

    # ============ LOAD CSV ============
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    # ============ PRECOMPUTE HYBRID FEATURES ============
    if use_hand:
        logger.info("Generating hand_feats, sentiment and num_ent for hybrid...")

        # -- 1) hand-crafted features
        from src.features import HandCraftedFeaturizer
        from sklearn.preprocessing import StandardScaler
        fe = HandCraftedFeaturizer()
        train_h = fe.transform(train_df.text)
        val_h   = fe.transform(val_df.text)

        scaler = StandardScaler().fit(train_h)
        train_h = scaler.transform(train_h)
        val_h   = scaler.transform(val_h)

        # save scaler
        MODEL_DIR = Path(OUTPUT_DIR) / ".." / "models"
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        import joblib
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

        train_df["hand_feats"] = train_h.tolist()
        val_df["hand_feats"]   = val_h.tolist()

        # -- 2) sentiment on CPU, batched
        sent_pipe = pipeline("sentiment-analysis",
                             model="distilbert-base-uncased-finetuned-sst-2-english",
                             device=-1,
                             batch_size=32)
        train_s = [1.0 if x["label"] == "NEGATIVE" else 0.0
                   for x in sent_pipe(list(train_df.text))]
        val_s   = [1.0 if x["label"] == "NEGATIVE" else 0.0
                   for x in sent_pipe(list(val_df.text))]
        train_df["sentiment"] = train_s
        val_df["sentiment"]   = val_s

        # -- 3) num_ent via spaCy
        nlp = spacy.load("en_core_web_sm")
        train_ne = [len(nlp(txt[:300]).ents) for txt in train_df.text]
        val_ne   = [len(nlp(txt[:300]).ents) for txt in val_df.text]
        train_df["num_ent"] = train_ne
        val_df["num_ent"]   = val_ne

        # overwrite CSV so future runs reuse them
        train_df.to_csv(TRAIN_PATH, index=False)
        val_df.to_csv(VAL_PATH,   index=False)

        logger.info("Hybrid features added to CSVs.")

    # ============ DATASETS & DATALOADERS ============
    train_ds = NewsDataset(train_df, use_hand_feats=use_hand)
    val_ds   = NewsDataset(val_df,   use_hand_feats=use_hand)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=2  # can be >0 now, no CUDA in dataset
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=2
    )

    # ============ MODEL ============
    model = get_model(args.model)
    model.to(device)
    logger.info(f"Model '{args.model}' params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    weights = compute_class_weights(train_df)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    train_loss_hist,  val_loss_hist  = [], []
    train_f1_hist,    val_f1_hist    = [], []

    save_path = OUTPUT_DIR / f"{args.model}_best.pt"

    # ============ TRAIN-VAL LOOP ============
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labels= batch["labels"].to(device)

            if use_hand: 
              hand   = batch["hand_feats"].to(device)      # [B, HAND_FDIM] 
              sent   = batch["sentiment"].to(device)       # [B, 1] 
              nent   = batch["num_ent"].to(device)         # [B, 1] 
              logits = model( 
                      input_ids=ids, 
                      attention_mask=mask, 
                      hand_feats=hand, 
                      sentiment=sent, 
                      num_ent=nent
            )
            else:
                logits = model(input_ids=ids, attention_mask=mask)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        epoch_train_loss = running / len(train_loader)
        train_loss_hist.append(epoch_train_loss)
        

        logger.info(f"[{args.model}] Epoch {epoch}/{epochs} train loss: {running/len(train_loader):.4f}")

        # val
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            val_running = 0.0
            for batch in val_loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                labels= batch["labels"].to(device)

                if use_hand:
                    hand = batch["hand_feats"].to(device)
                    sent = batch["sentiment"].to(device).unsqueeze(1)
                    ne   = batch["num_ent"].to(device).unsqueeze(1)
                    logits = model(
                        input_ids=ids,
                        attention_mask=mask,
                        hand_feats=hand,
                        sentiment=sent,
                        num_ent=ne
                    )
                else:
                    logits = model(input_ids=ids, attention_mask=mask)
                val_loss = loss_fn(logits, labels)
                val_running += val_loss.item()

                preds = torch.argmax(logits, dim=1)
                all_p.extend(preds.cpu().numpy())
                all_l.extend(labels.cpu().numpy())

        rep = classification_report(
            all_l, all_p,
            target_names=["real", "fake"],
            digits=4,
            output_dict=True
        )
        f1f = rep["fake"]["f1-score"]
        acc = (np.array(all_p)==np.array(all_l)).mean()
        logger.info(f"[{args.model}] Val acc={acc:.4f}, fake-F1={f1f:.4f}")
        
        val_loss_hist.append(val_running / len(val_loader))
        val_f1_hist.append(f1f)

        if f1f > best_f1:
            best_f1 = f1f
            torch.save(model.state_dict(), save_path)
            logger.info(f"[{args.model}] Saved best to {save_path.name}")


# -------------------Drawing a graph---------------------------
    # ────────────────────── SAVE CURVES ─────────────────────────
    model_out_dir = OUTPUT_DIR / args.model
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



if __name__ == "__main__":
    main()
