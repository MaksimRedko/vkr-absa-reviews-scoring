from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from configs.configs import config
from src.schemas.models import Candidate, ScoredCandidate
from src.stages.contracts import ScoringStage


class KeyBERTScorer(ScoringStage):
    def __init__(self, model: SentenceTransformer | None = None):
        self.model = model or SentenceTransformer(config.models.encoder_path)
        self.extractor_mode: str = str(getattr(config.discovery, "extractor", "ngram"))
        self.cosine_threshold: float = config.discovery.cosine_threshold
        self.keybert_top_k: int = config.discovery.keybert_top_k
        self.mmr_lambda: float = config.discovery.mmr_lambda
        self.mmr_top_k: int = config.discovery.mmr_top_k
        self.use_mmr: bool = self.extractor_mode == "ngram"
        self._emb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._emb_cache_max = int(
            getattr(config.discovery, "embedding_cache_max", 10000) or 0
        )

    # ------------------------------------------------------------------
    # Кэш эмбеддингов по строке (LRU)
    # ------------------------------------------------------------------
    def _embed_strings_cached(self, texts: List[str]) -> dict[str, np.ndarray]:
        if not texts:
            return {}
        if self._emb_cache_max <= 0:
            vecs = self.model.encode(texts, show_progress_bar=False)
            return {t: np.asarray(v, dtype=np.float32) for t, v in zip(texts, vecs)}

        out: dict[str, np.ndarray] = {}
        missing: List[str] = []
        for t in texts:
            if t in self._emb_cache:
                self._emb_cache.move_to_end(t)
                out[t] = self._emb_cache[t]
            else:
                missing.append(t)
        if missing:
            new_vecs = self.model.encode(missing, show_progress_bar=False)
            for t, v in zip(missing, new_vecs):
                arr = np.asarray(v, dtype=np.float32)
                self._emb_cache[t] = arr
                self._emb_cache.move_to_end(t)
                while len(self._emb_cache) > self._emb_cache_max:
                    self._emb_cache.popitem(last=False)
            for t in missing:
                out[t] = self._emb_cache[t]
        return out

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_text(candidate: Candidate) -> str:
        return str(candidate.source_span or candidate.span)

    def score_and_select(
        self, candidates: List[Candidate]
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        # Детерминированный порядок устраняет дрейф между запусками.
        sentences_unique = sorted({c.sentence for c in candidates})
        sent_to_emb = self._embed_strings_cached(sentences_unique)

        spans_unique = sorted({self._candidate_text(c) for c in candidates})
        span_to_emb = self._embed_strings_cached(spans_unique)

        by_sentence: dict[str, list[Candidate]] = {}
        for c in candidates:
            by_sentence.setdefault(c.sentence, []).append(c)

        results: list[ScoredCandidate] = []
        for sent, cands in by_sentence.items():
            scored = self._score_candidates(cands, sent_to_emb[sent], span_to_emb)
            selected = scored if not self.use_mmr else self._mmr(scored)
            results.extend(selected)

        return results

    # ------------------------------------------------------------------
    # Косинусный скоринг
    # ------------------------------------------------------------------
    def _score_candidates(
        self,
        candidates: List[Candidate],
        sent_emb: np.ndarray,
        span_to_emb: dict[str, np.ndarray],
    ) -> List[ScoredCandidate]:
        scored: list[ScoredCandidate] = []
        sent_vec = sent_emb.reshape(1, -1)

        for c in candidates:
            c_emb = span_to_emb[self._candidate_text(c)]
            sim = cosine_similarity(c_emb.reshape(1, -1), sent_vec)[0, 0]

            if sim < self.cosine_threshold:
                continue

            scored.append(
                ScoredCandidate(
                    span=c.span,
                    score=float(sim),
                    sentence=c.sentence,
                    embedding=c_emb,
                    review_id=getattr(c, "review_id", ""),
                    candidate_id=getattr(c, "candidate_id", ""),
                    source_span=self._candidate_text(c),
                )
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: self.keybert_top_k]

    # ------------------------------------------------------------------
    # MMR-диверсификация
    # ------------------------------------------------------------------
    def _mmr(self, scored: List[ScoredCandidate]) -> List[ScoredCandidate]:
        if len(scored) <= 1:
            return scored

        selected: list[ScoredCandidate] = [scored[0]]
        remaining = list(scored[1:])

        while remaining and len(selected) < self.mmr_top_k:
            best_idx = -1
            best_mmr = -float("inf")

            sel_embs = np.stack([s.embedding for s in selected])

            for i, cand in enumerate(remaining):
                relevance = cand.score
                max_sim_to_selected = cosine_similarity(
                    cand.embedding.reshape(1, -1), sel_embs
                ).max()
                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1 - self.mmr_lambda) * max_sim_to_selected
                )
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from src.stages.extraction import CandidateExtractor

    extractor = CandidateExtractor()
    scorer = KeyBERTScorer()

    test_text = "Экран шикарный, но батарея сдохла через день. Доставка быстрая."

    print(f"Текст: {test_text!r}\n")
    candidates = extractor.extract(test_text)
    print(f"Кандидатов после морфо-фильтра: {len(candidates)}")
    for c in candidates:
        print(f"  {c.span!r:30s} | sentence={c.sentence!r}")

    print()
    scored = scorer.score_and_select(candidates)
    print(f"Отобрано после KeyBERT + MMR: {len(scored)}")
    for s in scored:
        print(f"  {s.span!r:30s} score={s.score:.4f} | sentence={s.sentence!r}")
