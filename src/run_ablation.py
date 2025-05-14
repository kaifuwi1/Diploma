# run_ablation.py  (Colab / обычный python)

import csv, json, pathlib, subprocess, sys

PROJECT = pathlib.Path("/content/drive/MyDrive/FakeNewNet")
EVAL    = PROJECT / "src/evaluate.py"        # путь к evaluate.py
OUT_CSV = PROJECT / "src/outputs/ablation.csv"
MODELS  = ["baseline", "bigru", "cnn", "hybrid"]

HEADER  = [
    "model", "accuracy", "macro_F1",
    "prec_real", "rec_real", "f1_real",
    "prec_fake", "rec_fake", "f1_fake",
    "roc_auc", "pr_auc"
]

rows = []
for m in MODELS:
    ckpt = PROJECT / f"src/outputs/{m}_best.pt"
    if not ckpt.exists():
        print(f"⚠️  {m}: чекпойнт не найден, пропускаю");  continue

    out = subprocess.check_output(
        [sys.executable, str(EVAL),
         "--model", m,
         "--ckpt",  str(ckpt)],
        text=True
    )

    # Берём JSON-строку с метриками
    metrics = json.loads(next(l for l in out.splitlines() if l.lstrip().startswith("{")))
    rows.append([
        m,
        metrics["accuracy"], metrics["macro_F1"],
        metrics["precision_real"], metrics["recall_real"], metrics["f1_real"],
        metrics["precision_fake"], metrics["recall_fake"], metrics["f1_fake"],
        metrics["roc_auc"], metrics["pr_auc"],
    ])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    csv.writer(f).writerows([HEADER] + rows)

print("✅ ablation.csv готов  →", OUT_CSV)
