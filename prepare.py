import os
import json
import pandas as pd
import re
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_TOKENS = 256
BASE_DIRS = ["fakenewsnet_dataset/gossipcop", "fakenewsnet_dataset/politifact"]

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip().lower()

def extract_news(dataset):
    records = []
    for label in ["fake", "real"]:
        root = os.path.join(dataset, label)
        if not os.path.exists(root):
            continue
        for item in tqdm(os.listdir(root), desc=f"{dataset}/{label}"):
            folder = os.path.join(root, item)
            json_path = os.path.join(folder, "news content.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = clean_text(data.get("title", ""))
                    text = clean_text(data.get("text", ""))
                    tokens = tokenizer.tokenize(text)
                    if len(tokens) > MAX_TOKENS or not title or not text:
                        continue
                    records.append({
                        "id": item,
                        "dataset": dataset,
                        "title": title,
                        "text": text,
                        "label": label
                    })
            except Exception as e:
                print(f"⚠️ Failed to read {json_path}: {e}")
    return records

def main():
    all_records = []
    for dataset in BASE_DIRS:
        all_records += extract_news(dataset)

    df = pd.DataFrame(all_records)
    os.makedirs("data", exist_ok=True)

    # Stratified split
    train_val, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.125, stratify=train_val["label"], random_state=42)  # 0.125 * 0.8 = 0.1

    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("✅ Saved train/val/test to data/")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
