# src/models/bigru_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import PRETRAINED_MODEL_NAME, FREEZE_LAYERS


class DistilBERT_BiGRU(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()

        # 🔥 базовая модель
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)

        # 🔒 заморозим первые N слоёв
        for name, param in self.bert.named_parameters():
            layer_num = None
            if "transformer.layer." in name:
                layer_num = int(name.split("transformer.layer.")[1].split(".")[0])
            if layer_num is not None and layer_num < FREEZE_LAYERS:
                param.requires_grad = False

        hidden = self.bert.config.hidden_size  # обычно 768

        # BiGRU
        self.bigru = nn.GRU(
            input_size=hidden,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # attention-голова на выходе GRU
        self.attn = nn.Linear(256 * 2, 1)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256 * 2, num_labels)

    def forward(self, input_ids, attention_mask, return_att: bool = False, hand_feats=None):
        """
        Args:
            input_ids:       [B, L]
            attention_mask:  [B, L]
            return_att:      True -> возвращаем (logits, att_weights), иначе -> logits
        """

        # 1) BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_out.last_hidden_state           # [B, L, 768]

        # 2) BiGRU
        gru_out, _ = self.bigru(seq_out)               # [B, L, 512]

        # 3) Attention на токены
        # logits для каждого токена
        attn_logits = self.attn(gru_out).squeeze(-1)   # [B, L]
        # маскируем паддинги
        attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, L]

        # контекстный вектор
        context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)  # [B, 512]

        # 4) Классификатор
        logits = self.classifier(self.dropout(context))  # [B, num_labels]

        if return_att:
            return logits, attn_weights

        return logits
