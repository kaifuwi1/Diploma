# src/dataset.py

import ast
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import PRETRAINED_MODEL_NAME, MAX_LEN


class NewsDataset(Dataset):
    def __init__(self, df, use_hand_feats: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.use_hand_feats = use_hand_feats

        if self.use_hand_feats and "hand_feats" not in self.df.columns:
            raise ValueError("`use_hand_feats=True`, но нет колонки 'hand_feats'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row.text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item: Dict[str, Any] = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(0 if row.label=="real" else 1, dtype=torch.long),
        }

        if self.use_hand_feats:
            raw = row.hand_feats
            if isinstance(raw, str):
                raw = ast.literal_eval(raw)
            item["hand_feats"] = torch.tensor(np.asarray(raw, dtype=np.float32))

            item["sentiment"] = torch.tensor([row.sentiment], dtype=torch.float)
            item["num_ent"]   = torch.tensor([row.num_ent],   dtype=torch.float)

        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {
            "input_ids":      torch.stack([x["input_ids"]      for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels":         torch.stack([x["labels"]         for x in batch]),
        }
        if self.use_hand_feats:
            out["hand_feats"]  = torch.stack([x["hand_feats"]  for x in batch])
            out["sentiment"]   = torch.stack([x["sentiment"]   for x in batch])
            out["num_ent"]     = torch.stack([x["num_ent"]     for x in batch])
        return out
