# ── src/benchmark.py ────────────────────────────────────────────────
import time, csv, pathlib, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path("/content/drive/MyDrive/FakeNewNet")
sys.path.append(str(ROOT))

from src.config   import TEST_PATH, DEVICE, HAND_FDIM
from src.dataset  import NewsDataset
from src.models   import get_model

MODELS  = ["baseline", "bigru", "cnn", "hybrid"]
N_ART   = 1000
BATCH   = 16
OUT_CSV = ROOT / "src/outputs/benchmark.csv"


def _add_dummy_feats(df: pd.DataFrame):
    """Добавляем нули-заглушки, если колонок нет."""
    if "hand_feats" not in df.columns:
        zeros = [ [0.0]*HAND_FDIM ] * len(df)
        df["hand_feats"] = zeros
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    if "num_ent" not in df.columns:
        df["num_ent"] = 0.0
    return df


def _measure(model, loader, device):
    model.eval()
    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            if "hand_feats" in batch:           # hybrid
                logits = model(
                    ids, mask,
                    hand_feats=batch["hand_feats"].to(device),
                    sentiment=batch["sentiment"].to(device),
                    num_ent=batch["num_ent"].to(device),
                )
            else:
                logits = model(ids, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) / len(loader.dataset) * 1000  # ms


def main():
    print("📊 Benchmark …")
    rows = []

    df_all = pd.read_csv(TEST_PATH).sample(N_ART, random_state=42).reset_index(drop=True)

    for name in MODELS:
        ckpt = ROOT / f"src/outputs/{name}_best.pt"
        if not ckpt.exists():
            print(f"⚠️  {ckpt.name} not found — skip");  continue

        use_hand = name == "hybrid"
        df = _add_dummy_feats(df_all.copy()) if use_hand else df_all

        ds = NewsDataset(df, use_hand_feats=use_hand)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        collate_fn=ds.collate_fn, num_workers=0)

        for dev in ["cpu", "cuda"]:
            if dev == "cuda" and not torch.cuda.is_available():
                continue

            device = torch.device(dev)
            model  = get_model(name).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            t_ms = _measure(model, dl, device)

            rows.append([name, dev, round(t_ms, 2)])
            print(f"{name:8} {dev}: {t_ms:.2f} ms/article")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows([["model","device","ms_per_article"]] + rows)
    print("✅ benchmark.csv →", OUT_CSV)


if __name__ == "__main__":
    main()
