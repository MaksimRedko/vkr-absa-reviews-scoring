from __future__ import annotations

from collections import OrderedDict

import torch

from src.stages.nli_persistent_cache import CacheStats, PersistentNliCache, build_model_signature, cached_pairs_from_strings
from src.stages.sentiment import SentimentEngine


def test_persistent_cache_deduplicates_text_rows(tmp_path):
    cache = PersistentNliCache(
        path=tmp_path / "nli_cache.sqlite3",
        model_signature="model-a",
        enabled=True,
    )
    pairs = cached_pairs_from_strings(
        ["один и тот же отзыв", "один и тот же отзыв"],
        ["аспект хороший", "аспект плохой"],
    )
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=torch.float32).numpy()
    written = cache.store_many(pairs, logits)

    assert written == 2
    assert cache.count_rows() == 2
    assert cache.count_text_rows() == 3

    cache.close()


def _build_dummy_engine(cache_path, *, model_signature: str):
    engine = SentimentEngine.__new__(SentimentEngine)
    engine.num_labels = 3
    engine.batch_size = 16
    engine._use_onnx = False
    engine.model = None
    engine._nli_cache = OrderedDict()
    engine._nli_cache_max = 0
    engine._cache_stats = CacheStats()
    engine._persistent_cache = PersistentNliCache(
        path=cache_path,
        model_signature=model_signature,
        enabled=True,
    )
    engine._uncached_calls = 0

    def _fake_forward(premises, hypotheses):
        engine._uncached_calls += 1
        rows = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            base = float(len(premise) + len(hypothesis))
            rows.append([base, base + 1.0, base + 2.0])
        return torch.tensor(rows, dtype=torch.float32)

    engine._uncached_forward_logits_tensor = _fake_forward
    return engine


def test_sentiment_engine_uses_persistent_cache_between_instances(tmp_path):
    cache_path = tmp_path / "nli_cache.sqlite3"
    signature = build_model_signature(
        backend="pytorch",
        model_path="model-path",
        tokenizer_path="model-path",
        id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
        num_labels=3,
    )
    premises = ["отзыв про катушку", "отзыв про доставку"]
    hypotheses = ["катушка хорошая", "доставка хорошая"]

    cold = _build_dummy_engine(cache_path, model_signature=signature)
    cold_tensor = SentimentEngine._forward_logits_tensor(cold, premises, hypotheses)
    cold_stats = cold.get_cache_stats()

    assert cold._uncached_calls == 1
    assert cold_stats["persistent_hits"] == 0
    assert cold_stats["misses"] == 2
    assert cold_stats["writes"] == 2
    assert cold_stats["persistent_rows"] == 2
    cold._persistent_cache.close()

    warm = _build_dummy_engine(cache_path, model_signature=signature)
    warm_tensor = SentimentEngine._forward_logits_tensor(warm, premises, hypotheses)
    warm_stats = warm.get_cache_stats()

    assert warm._uncached_calls == 0
    assert warm_stats["persistent_hits"] == 2
    assert warm_stats["misses"] == 0
    assert warm_stats["writes"] == 0
    assert torch.equal(cold_tensor.cpu(), warm_tensor.cpu())
    warm._persistent_cache.close()


def test_persistent_cache_separates_different_model_signatures(tmp_path):
    cache_path = tmp_path / "nli_cache.sqlite3"
    cache_a = PersistentNliCache(
        path=cache_path,
        model_signature="model-a",
        enabled=True,
    )
    pairs = cached_pairs_from_strings(["один отзыв"], ["одна гипотеза"])
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    cache_a.store_many(pairs, logits)

    cache_b = PersistentNliCache(
        path=cache_path,
        model_signature="model-b",
        enabled=True,
    )
    found = cache_b.lookup_many(pairs)

    assert found == {}

    cache_a.close()
    cache_b.close()
