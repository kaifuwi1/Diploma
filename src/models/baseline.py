# src/models/baseline.py
import sys
sys.path.append("/content/drive/MyDrive/FakeNewNet")

import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import PRETRAINED_MODEL_NAME

class DistilBERTBaseline(nn.Module):
    def __init__(self, num_labels=2):
        super(DistilBERTBaseline, self).__init__()
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, hand_feats=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
