# src/models/__init__.py

from .baseline import DistilBERTBaseline
from .bigru_head import DistilBERT_BiGRU
from .cnn_head   import CNNHead
from .hybrid     import HybridModel   # ← NEW



_MODEL_REGISTRY = {
    "baseline": DistilBERTBaseline,
    "bigru":    DistilBERT_BiGRU,
    "cnn":      CNNHead,
    "hybrid":   HybridModel         # ← регистрация гибрид
}


def get_model(name: str):
    try:
        return _MODEL_REGISTRY[name]()    # создать экземпляр
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(_MODEL_REGISTRY.keys())}")
