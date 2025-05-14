# src/models/cnn_head.py
import torch
import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")
from src.config import PRETRAINED_MODEL_NAME, FREEZE_LAYERS


class CNNHead(nn.Module):
    """DistilBERT (фриз ±3 слоя)  → Conv1D{k=3,4,5}@128 → MaxPool → Linear(2)"""

    def __init__(self, num_labels: int = 2, freeze_layers: int = 3, hidden_size: int = 768):
        super().__init__()

        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
        # частично замораживаем
        for name, param in self.bert.named_parameters():
            layer_num = (
                int(name.split(".")[2]) if name.startswith("transformer.layer") else None
            )
            if layer_num is not None and layer_num < freeze_layers:
                param.requires_grad_(False)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=128,
                    kernel_size=k,
                    padding="same",
                )
                for k in (3, 4, 5)
            ]
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128 * 3, num_labels)

    def forward(self, input_ids, attention_mask, return_att: bool = False, hand_feats=None):
        """
        Args:
            input_ids:      [B, L]
            attention_mask: [B, L]
            return_att:     если True — возвращаем (logits, None)
        """
        # 1) BERT → last_hidden_state (B, L, H)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state              # (B, L, H)
        x = x.transpose(1, 2)                      # (B, H, L) для Conv1d

        # 2) Conv+MaxPool
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))                # (B, 128, L)
            p = torch.max(c, dim=2).values         # (B, 128)
            pooled.append(p)

        feat = torch.cat(pooled, dim=1)            # (B, 128*3)
        feat = self.dropout(feat)
        logits = self.fc(feat)                     # (B, num_labels)

        if return_att:
            # в CNNHead нет реальных attention-весов, возвращаем None
            return logits, None

        return logits
