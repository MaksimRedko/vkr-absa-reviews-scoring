from __future__ import annotations

import re
from typing import List, Literal, Optional

import pymorphy3

from configs.configs import config
from src.schemas.models import Candidate
from src.stages.contracts import ExtractionStage
from src.stages.parsing import DependencyParser

_morph = pymorphy3.MorphAnalyzer()

STOP_TOKENS: set[str] = {
    "достоинства", "достоинство", "недостатки", "недостаток",
    "минусов", "минус", "плюсы", "плюс", "комментарий",
    "рекомендую", "спасибо",
    'ребенок','ребенку',
    'сыну',
    'дочке',
    'мужу',
    'жене',
    'сын', 'дочь', 'муж', 'жена', 'подарок', 'день', 'рождение',
    'брат', 'сестра',
    'дочка',
    'друг','другу',
}


class CandidateExtractor(ExtractionStage):
    def __init__(
        self,
        ngram_range: tuple[int, int] | None = None,
        min_word_length: int | None = None,
    ):
        _nr = ngram_range or tuple(config.discovery.ngram_range)
        self.ngram_min: int = _nr[0]
        self.ngram_max: int = _nr[1]
        self.min_word_length: int = (
            min_word_length
            if min_word_length is not None
            else config.discovery.min_word_length
        )
        self.dependency_filter_enabled: bool = bool(
            getattr(config.discovery, "dependency_filter_enabled", False)
        )
        self.dependency_filter_mode: Literal["all_heads", "aspect_roles"] = str(
            getattr(config.discovery, "dependency_filter_mode", "aspect_roles")
        )
        self.last_filter_stats: dict[str, object] = {}
        self._dependency_parser: DependencyParser | None = None
        if self.dependency_filter_enabled:
            fallback_models = list(
                getattr(
                    config.discovery,
                    "dependency_spacy_fallback_models",
                    ["ru_core_news_md", "ru_core_news_sm"],
                )
            )
            self._dependency_parser = DependencyParser(
                preferred_model=str(
                    getattr(config.discovery, "dependency_spacy_model", "ru_core_news_lg")
                ),
                fallback_models=fallback_models,
                include_root_verbs=bool(
                    getattr(config.discovery, "dependency_include_root_verbs", True)
                ),
                include_root_adjs=bool(
                    getattr(config.discovery, "dependency_include_root_adjs", True)
                ),
            )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = self._clean(raw_text)
        sentences = self._split_sentences(cleaned_text)
        allowed_filter_lemmas, parse_meta = self._get_allowed_filter_lemmas(cleaned_text)

        results: list[Candidate] = []
        before_filter = 0
        for sent in sentences:
            candidates = self._candidates_from_sentence(sent)
            before_filter += len(candidates)
            if allowed_filter_lemmas is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if self._candidate_matches_heads(candidate, allowed_filter_lemmas)
                ]
            results.extend(candidates)
        self.last_filter_stats = {
            "filter_enabled": bool(allowed_filter_lemmas is not None),
            "filter_mode": self.dependency_filter_mode,
            "allowed_head_lemmas": sorted((parse_meta or {}).get("head_lemmas", [])),
            "allowed_filter_lemmas": sorted(allowed_filter_lemmas or []),
            "parser_model": (parse_meta or {}).get("model_name"),
            "parser_available": bool((parse_meta or {}).get("parser_available", False)),
            "parse_failed": bool((parse_meta or {}).get("parse_failed", False)),
            "candidates_before_filter": int(before_filter),
            "candidates_after_filter": int(len(results)),
        }
        return results

    # ------------------------------------------------------------------
    # 1. Предобработка
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        text = str(text).replace("\\n", " ")
        text = re.sub(r"(?<=[.,!?;:])(?=[^\s])", " ", text)
        text = re.sub(r"[^\w\s.,!?;:\-]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences if sentences else [text]

    # ------------------------------------------------------------------
    # 2–4. Генерация N-грамм → морфо-фильтр → дедупликация
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(sentence: str) -> List[str]:
        raw = sentence.lower().split()
        tokens = [re.sub(r"[^\w\-]", "", t) for t in raw]
        return [t for t in tokens if t]

    def _get_allowed_filter_lemmas(
        self,
        text: str,
    ) -> tuple[Optional[set[str]], Optional[dict[str, object]]]:
        if not self.dependency_filter_enabled or self._dependency_parser is None:
            return None, None
        parsed = self._dependency_parser.parse(text)
        filter_lemmas: set[str]
        if self.dependency_filter_mode == "all_heads":
            filter_lemmas = set(parsed.head_lemmas)
        else:
            filter_lemmas = set(parsed.aspect_role_lemmas)

        parse_meta = {
            "head_lemmas": set(parsed.head_lemmas),
            "aspect_role_lemmas": set(parsed.aspect_role_lemmas),
            "model_name": parsed.model_name,
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
        }
        if parsed.parse_failed or (not parsed.parser_available) or (not filter_lemmas):
            return None, parse_meta
        return filter_lemmas, parse_meta

    @staticmethod
    def _lemma(token: str) -> str:
        parses = _morph.parse(token)
        if not parses:
            return token.lower()
        return str(parses[0].normal_form or token).lower()

    def _candidate_matches_heads(
        self,
        candidate: Candidate,
        allowed_head_lemmas: set[str],
    ) -> bool:
        candidate_tokens = self._tokenize(candidate.span)
        candidate_lemmas = {self._lemma(token) for token in candidate_tokens}
        return bool(candidate_lemmas & allowed_head_lemmas)

    def _candidates_from_sentence(self, sentence: str) -> List[Candidate]:
        tokens = self._tokenize(sentence)
        seen: set[str] = set()
        candidates: list[Candidate] = []

        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(tokens) - n + 1):
                gram_tokens = tokens[i : i + n]

                if any(len(t) < self.min_word_length for t in gram_tokens):
                    continue

                span = " ".join(gram_tokens)

                if span in seen:
                    continue

                if any(t in STOP_TOKENS for t in gram_tokens):
                    continue

                if not self._pass_morph_filter(gram_tokens):
                    continue

                seen.add(span)
                candidates.append(
                    Candidate(
                        span=span,
                        sentence=sentence,
                        token_indices=(i, i + n),
                    )
                )

        return candidates

    # ------------------------------------------------------------------
    # 3. Морфологический фильтр
    # ------------------------------------------------------------------
    _BIGRAM_POS_PATTERNS: set[tuple[str, str]] = {
        ("ADJ", "NOUN"),
        ("NOUN", "NOUN"),
        ("NOUN", "ADJ"),
    }

    def _pass_morph_filter(self, tokens: list[str]) -> bool:
        if len(tokens) == 1:
            return self._is_noun_or_oov(tokens[0])

        pos_tags = [self._best_pos(t) for t in tokens]

        if any(p == "OOV" for p in pos_tags):
            return True

        return (pos_tags[0], pos_tags[1]) in self._BIGRAM_POS_PATTERNS

    @staticmethod
    def _best_pos(token: str) -> str:
        parses = _morph.parse(token)
        if not any(p.is_known for p in parses):
            return "OOV"
        for p in parses:
            if "NOUN" in p.tag:
                return "NOUN"
        for p in parses:
            if "ADJF" in p.tag or "ADJS" in p.tag:
                return "ADJ"
        return str(parses[0].tag.POS or "X")

    @staticmethod
    def _is_noun_or_oov(token: str) -> bool:
        parses = _morph.parse(token)
        if not any(p.is_known for p in parses):
            return True
        return any("NOUN" in p.tag for p in parses)


if __name__ == "__main__":
    extractor = CandidateExtractor()

    test_texts = [
        "Достоинства: Отлично очищают экран, не оставляют разводов.",
        "норм пришло быстро качество ок",
        "Сдохла батарея через неделю, просто ужас",
        "Rick Owens качество огонь",
    ]

    for sample_text in test_texts:
        print(f"\n--- Текст: {sample_text!r} ---")
        sample_candidates = extractor.extract(sample_text)
        for c in sample_candidates:
            print(f"  span={c.span!r:30s}  sentence={c.sentence!r}")
