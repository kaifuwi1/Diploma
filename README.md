# Diploma

# Neural model for semantic analysis of news articles – Bachelor Diploma (Astana IT University)

A compact, fully-reproducible pipeline that fine-tunes **DistilBERT** and several light-weight heads (BiGRU, CNN, Hybrid) to classify news articles as **Real vs Fake**, highlight suspicious tokens, and expose word-cloud + HTML explanations – all runnable in Google Colab on a free GPU.

# Prerequisites
Before running the application, ensure you have the following set up:
- Python 3.10.6
- Parsed dataset from https://github.com/KaiDMML/FakeNewsNet/tree/master

#

| Stage | Output | Key files |
|-------|--------|-----------|
| Data prep | `data/train val test.csv` stratified splits | `prepare.py`, `src/dataset.py` |
| Baselines & heads | `*_best.pt` checkpoints + loss/F1 curves | `src/models/` + `train.py` |
| Ablations | `outputs/ablation.csv` | `evaluate.py` + `scripts/ablate.py` |
| Explainability | `outputs/cloud.png`, `outputs/explain.html` | `src/viz.py`, `explain_sample.py` |
| Benchmark | `outputs/benchmark.txt` | `benchmark.py` |


## Directory layout:
├── data/ # CSV splits (real/fake)\
├── src/\
│       ├── config.py # paths & hyper-params\
│       ├── dataset.py # NewsDataset + collate\
│       ├── models/ # baseline, bigru_head, cnn_head, hybrid\
│       ├── features.py # 12 hand-crafted features\
│       ├── viz.py # IntegratedGradients, word-cloud, HTML\
│       ├── train.py # generic trainer\
│       ├── train_hybrid.py # hybrid-specific (α-balanced FocalLoss)\
│       ├── evaluate.py # metrics + confusion matrix\
│       └── benchmark.py # 1 000-article latency test\
├── outputs/ # checkpoints, PNGs, CSVs – auto-created\
└── prepare.py # one-click data download / split\


## Quick start (Colab)

```bash
# 0) clone with LFS to keep checkpoints small
!git clone https://github.com/kaifuwi1/Diploma.git
%cd Diploma

# 1) set up environment (≈3 min)
!pip install -r requirements.txt

# 2) train any model
!python src/train.py   --model baseline --epochs 3
!python src/train.py   --model bigru    --epochs 5
!python src/train.py   --model cnn      --epochs 5
!python src/train.py   --model hybrid   --epochs 4  # adds sentiment & NER feats

# 3) evaluate + confusion matrix
!python src/evaluate.py  --model hybrid \
        --ckpt src/outputs/hybrid_best.pt

# 4) word-cloud + HTML highlight for a single article
!python src/explain_sample.py \
        --ckpt src/outputs/hybrid_best.pt \
        --text examples/blake_lively.txt

# 5) latency benchmark (CPU & GPU)
!python src/benchmark.py

```

## Re-using the scaler
When you first run train.py --model hybrid, the script computes 12 statistical features fits a StandardScaler dumps it to src/models/scaler.pkl. During inference the scaler is auto-loaded – no manual steps required.
