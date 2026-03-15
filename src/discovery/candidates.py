from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import pymorphy3

from configs.configs import config

_morph = pymorphy3.MorphAnalyzer()

STOP_TOKENS: set[str] = {
    "достоинства", "достоинство", "недостатки", "недостаток",
    "минусов", "минус", "плюсы", "плюс", "комментарий",
    "рекомендую", "спасибо",
    'ребенок',
    'ребенку',
    'сыну',
    'дочке',
    'мужу',
    'жене',
    'сын', 'дочь', 'муж', 'жена', 'подарок', 'день', 'рождение',
    'брат', 'сестра',
    'дочка',
    'друг',
    'другу',


}


@dataclass
class Candidate:
    span: str
    sentence: str
    token_indices: tuple[int, int]


class CandidateExtractor:
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

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def extract(self, raw_text: str) -> List[Candidate]:
        text = self._clean(raw_text)
        sentences = self._split_sentences(text)

        results: list[Candidate] = []
        for sent in sentences:
            candidates = self._candidates_from_sentence(sent)
            results.extend(candidates)
        return results

    # ------------------------------------------------------------------
    # 1. Предобработка
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
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

    for text in test_texts:
        print(f"\n--- Текст: {text!r} ---")
        candidates = extractor.extract(text)
        for c in candidates:
            print(f"  span={c.span!r:30s}  sentence={c.sentence!r}")
