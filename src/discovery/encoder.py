from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_DISCOVERY_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
DEFAULT_DISCOVERY_CACHE_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_DISCOVERY_LOCAL_DIR = DEFAULT_DISCOVERY_CACHE_DIR / "discovery_sbert_large_nlu_ru"
DEFAULT_DISCOVERY_EMBEDDING_DIM = 1024


class DiscoveryEncoder:
    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_DISCOVERY_MODEL_NAME,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        cache_dir: str | Path = DEFAULT_DISCOVERY_CACHE_DIR,
        local_dir: str | Path = DEFAULT_DISCOVERY_LOCAL_DIR,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.model_name_or_path = str(model_name_or_path)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.cache_dir = Path(cache_dir)
        self.local_dir = Path(local_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = DEFAULT_DISCOVERY_EMBEDDING_DIM

        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        self._ensure_model()

        normalized_texts = [str(text) for text in texts]
        batches: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(normalized_texts), self.batch_size):
                batch_texts = normalized_texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self._model(**encoded)
                pooled = self._mean_pool(
                    last_hidden_state=outputs.last_hidden_state,
                    attention_mask=encoded["attention_mask"],
                )
                pooled = F.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))

        matrix = np.concatenate(batches, axis=0)
        if matrix.shape != (len(texts), self.embedding_dim):
            raise RuntimeError(
                f"Unexpected embedding shape {matrix.shape}; "
                f"expected {(len(texts), self.embedding_dim)}."
            )
        return matrix

    def _ensure_model(self) -> None:
        if self.is_loaded:
            return

        model_source = self._resolve_model_source()

        if model_source == str(self.local_dir):
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                model_source,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    cache_dir=str(self.cache_dir),
                    local_files_only=True,
                )
                model = AutoModel.from_pretrained(
                    model_source,
                    cache_dir=str(self.cache_dir),
                    local_files_only=True,
                )
            except OSError:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                )
                model = AutoModel.from_pretrained(
                    model_source,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                )

        self._tokenizer = tokenizer
        self._model = model.to(self.device)
        self._model.eval()

        hidden_size = int(self._model.config.hidden_size)
        if hidden_size != DEFAULT_DISCOVERY_EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected hidden size {hidden_size}; "
                f"expected {DEFAULT_DISCOVERY_EMBEDDING_DIM}."
            )
        self.embedding_dim = hidden_size

    def _resolve_model_source(self) -> str:
        if self.local_dir.exists():
            return str(self.local_dir)
        return self.model_name_or_path

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
