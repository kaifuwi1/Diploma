# src/inference.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")

from src.config import TEST_PATH, MODEL_DIR, PRETRAINED_MODEL_NAME, MAX_LEN
from src.models.baseline import DistilBERTBaseline

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def predict():
    df = pd.read_csv(TEST_PATH)
    texts = df["text"].tolist()

    dataset = TestDataset(texts)
    loader = DataLoader(dataset, batch_size=16)

    model = DistilBERTBaseline()
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pt", map_location="cpu"))
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    df["prediction"] = ["real" if p == 0 else "fake" for p in predictions]
    df.to_csv("output.csv", index=False)
    print("✅ Saved predictions to output.csv")

if __name__ == "__main__":
    predict()
