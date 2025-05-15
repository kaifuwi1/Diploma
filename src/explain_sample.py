# src/explain_sample.py
import argparse, pathlib, sys, torch

# --- project imports --------------------------------------------------
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import DEVICE, PRETRAINED_MODEL_NAME, HAND_FDIM
from src.models import get_model
from src.viz   import get_token_importance, make_wordcloud, html_highlight
from transformers import AutoTokenizer
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="model checkpoint")
parser.add_argument("--text", required=True, help="path to plain-text file")
args = parser.parse_args()

# ---------- загрузка модели ------------------------------------------
model_name = pathlib.Path(args.ckpt).stem.split("_")[0]     # baseline_best.pt → baseline
tokenizer  = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
model      = get_model(model_name).to(DEVICE)
model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
model.eval()

# ---------- читаем текст ---------------------------------------------
text = pathlib.Path(args.text).read_text(encoding="utf-8")

# ---------- считаем P(fake) ------------------------------------------
enc = tokenizer(text, return_tensors="pt", truncation=True,
                padding="max_length", max_length=256).to(DEVICE)

if "hand_feats" in model.forward.__code__.co_varnames:      # HybridModel
    B = 1
    hand = torch.zeros(B, HAND_FDIM, device=DEVICE)
    sent = torch.zeros(B, 1,           device=DEVICE)
    ne   = torch.zeros(B, 1,           device=DEVICE)
    logits = model(enc["input_ids"], enc["attention_mask"],
                   hand_feats=hand, sentiment=sent, num_ent=ne)
else:                                                       # baseline / bigru / cnn
    logits = model(enc["input_ids"], enc["attention_mask"])

prob_fake = torch.softmax(logits, 1)[0, 1].item()           # float ∈ [0..1]

# ---------- IG → tokens importance ------------------------
tokens_imp = get_token_importance(model, tokenizer, text, DEVICE)

# ---------- визуализация ----------------------------------
out_dir = pathlib.Path("/content/drive/MyDrive/FakeNewNet/src/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
print("Result:", "fake" if prob_fake > 50 else "real", '\n')

make_wordcloud(tokens_imp, out_dir / f"{model_name}cloud.png")           # only fake-tokens
html_highlight(tokens_imp, out_dir / f"{model_name}explain.html", prob_fake=prob_fake)

print(f"✅  Outputs saved to {out_dir}\n    cloud.png\n    explain.html")
