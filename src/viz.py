# ── src/viz.py ─────────────────────────────────────────────────────────
"""
Explainability helpers
----------------------
* get_token_importance  — Integrated Gradients → [(token, importance)]
* make_wordcloud        — PNG-облако из |importance|
* html_highlight        — HTML-подсветка: красный → fake, зелёный → real
"""

from __future__ import annotations
import html as _html
from pathlib import Path
import unicodedata, re

import torch
try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "captum"])
    from captum.attr import LayerIntegratedGradients

from transformers import AutoTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import PRETRAINED_MODEL_NAME, HAND_FDIM


# ───────────────────────── helpers ────────────────────────────────────
def clean_text(txt: str) -> str:
    """Убираем битые символы типа Â€ etc., приводим к NFC."""
    txt = unicodedata.normalize("NFKC", txt)
    return re.sub(r"[^\x20-\x7E]", " ", txt)


def _merge_wordpieces(tokens: list[str], scores):
    """
    'inj', '##ure', '##s'  →  'injures'
    importance усредняем.
    """
    merged, merged_scores = [], []
    buffer, buf_score, cnt = "", 0.0, 0

    for tok, sc in zip(tokens, scores):
        if tok.startswith("##"):
            buffer += tok[2:]
            buf_score += sc
            cnt += 1
        else:
            if buffer:
                merged.append(buffer)
                merged_scores.append(buf_score / max(cnt, 1))
            buffer, buf_score, cnt = tok, sc, 1
    if buffer:
        merged.append(buffer)
        merged_scores.append(buf_score / max(cnt, 1))
    return merged, merged_scores


# ───────────────────────── core IG ────────────────────────────────────
def get_token_importance(model,
                         tokenizer: AutoTokenizer,
                         text: str,
                         device: str | torch.device = "cpu",
                         max_len: int = 256):
    """
    Returns list[(token, importance)].  importance>0 → вклад в fake.
    Работает с baseline / bigru / cnn / hybrid.
    """
    text = clean_text(text)
    model.to(device).eval()

    # --- временно переводим в train() для GRU, потом вернём как было ----
    was_training = model.training            # True / False
    model.train()                            # нужен для cuDNN-GRU backward

    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt"
    ).to(device)

    # embeddings-layer
    if hasattr(model, "bert"):
        embed_layer = model.bert.embeddings
    elif hasattr(model, "distilbert"):
        embed_layer = model.distilbert.embeddings
    else:
        raise RuntimeError("embeddings-layer not found")

    def forward_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Универсальная обёртка для LayerIntegratedGradients.
        Передаём ровно те дополнительные тензоры, которые
        реально предусмотрены сигнатурой модели.

        Возвращаем логиты класса «fake» (index 1).
        """
        sig = model.forward.__code__.co_varnames
        kwargs = dict(input_ids=input_ids,
                      attention_mask=attention_mask)

        if "hand_feats" in sig:
            B = input_ids.size(0)
            kwargs["hand_feats"] = torch.zeros(B, HAND_FDIM,
                                              device=input_ids.device)

        if "sentiment" in sig:
            B = input_ids.size(0)
            kwargs["sentiment"] = torch.zeros(B, 1,
                                              device=input_ids.device)

        if "num_ent" in sig:
            B = input_ids.size(0)
            kwargs["num_ent"] = torch.zeros(B, 1,
                                            device=input_ids.device)

        logits = model(**kwargs)          # [B, 2]
        return logits[:, 1]               # logit(fake)



    lig = LayerIntegratedGradients(forward_fn, embed_layer)
    with torch.enable_grad():
        attributions = lig.attribute(
            inputs=enc["input_ids"],
            baselines=torch.zeros_like(enc["input_ids"]),
            additional_forward_args=(enc["attention_mask"],),
            return_convergence_delta=False,
        )

    scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))

        # … после вычисления scores / tokens …
    # -------------------------------------
    if not was_training:
        model.eval()     # возвращаем eval, если так было изначально


    # drop [CLS]/[SEP]/[PAD]
    filt = [(t, s) for t, s in zip(tokens, scores)
            if t not in tokenizer.all_special_tokens]

    # склеиваем WordPieces
    toks, scs = _merge_wordpieces(*zip(*filt)) if filt else ([], [])
    return list(zip(toks, scs))


# ───────────────────────── word-cloud ─────────────────────────────────
def make_wordcloud(tokens_imp: list[tuple[str, float]],
                   out_png: str | Path,
                   top_k: int | None = 100):
    """
    Cloud из «fake»-токенов:
       • берём только importance > 0
       • размер слова ∝ importance
       • top_k – сколько самых влиятельных слов оставить (None = все)
    """
    # фильтруем токены, которые двигают к fake
    pos = [(tok, imp) for tok, imp in tokens_imp if imp > 0]
    if not pos:
        print("⚠️  нет позитивных токенов, облако не создано");  return

    # берём Top-K
    pos.sort(key=lambda x: x[1], reverse=True)
    if top_k:
        pos = pos[:top_k]

    freqs = {tok: imp for tok, imp in pos}          # importance > 0
    wc = WordCloud(width=600, height=300,
                   background_color="white",
                   colormap="Reds")                 # красная палитра
    wc.generate_from_frequencies(freqs)
    wc.to_file(str(out_png))
    # print(f"✅ fake-wordcloud saved → {out_png}")


# ───────────────────────── HTML highlight ─────────────────────────────
def html_highlight(tokens_imp: list[tuple[str, float]],
                   out_html: str | Path,
                   prob_fake: float | None = None):
    """
    Делает HTML-подсветку:
       • красный  → вклад в fake
       • зелёный  → вклад в real
       • если передать prob_fake, вверху появится заголовок P(fake)=…
    """
    if not tokens_imp:
        Path(out_html).write_text("<p>(empty)</p>", encoding="utf-8")
        return

    max_abs = max(abs(imp) for _, imp in tokens_imp) + 1e-9
    html_toks = []
    for tok, imp in tokens_imp:
        alpha = abs(imp) / max_abs
        color = f"rgba(255,0,0,{alpha:.2f})" if imp > 0 \
                else f"rgba(60,179,113,{alpha:.2f})"
        html_toks.append(
            f'<span style="background:{color}">{_html.escape(tok)}</span>'
        )

    header = ""
    if prob_fake is not None:
        header = f"<p><b>P(fake) = {prob_fake:.2%}</b></p>"

    Path(out_html).write_text(
        header + "<p>" + " ".join(html_toks) + "</p>", encoding="utf-8"
    )
    # print(f"✅ HTML highlight saved → {out_html}")


