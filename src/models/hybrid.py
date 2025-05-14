# src/models/hybrid.py
import torch
import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import PRETRAINED_MODEL_NAME, HAND_FDIM, FREEZE_LAYERS

class HybridModel(nn.Module):
    """DistilBERT-CLS ∥ hand_feats ∥ sentiment ∥ num_ent → MLP"""

    def __init__(self, num_labels: int = 2):
        super().__init__()
        # загружаем базовую модель DistilBERT
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)

        # частично замораживаем первые FREEZE_LAYERS слоёв
        for name, param in self.bert.named_parameters():
            if "transformer.layer." in name:
                layer_idx = int(name.split("transformer.layer.")[1].split(".")[0])
                if layer_idx < FREEZE_LAYERS:
                    param.requires_grad = False

        # размер входа в полносвязный слой: CLS(768) + HAND_FDIM + sentiment(1) + num_ent(1)
        in_dim = self.bert.config.hidden_size + HAND_FDIM + 1 + 1
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                hand_feats,
                sentiment,
                num_ent):
        """
        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            hand_feats:     [batch, HAND_FDIM]
            sentiment:      [batch, 1]
            num_ent:        [batch, 1]
        """
        # получаем эмбеддинг CLS
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]

        # конкатенируем все признаки
        x = torch.cat([cls_emb, hand_feats, sentiment, num_ent], dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
