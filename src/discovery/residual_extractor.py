from __future__ import annotations

import re
from dataclasses import dataclass

import pymorphy3

from src.schemas.models import ReviewInput
from src.stages.extraction import CandidateExtractor
from src.vocabulary.loader import AspectDefinition, Vocabulary

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)

@dataclass(slots=True)
class ResidualResult:
    review_id: str
    covered_phrases: list[str]
    covered_aspects: list[str]
    residual_phrases: list[str]


class ResidualExtractor:
    def __init__(
        self,
        *,
        candidate_extractor: CandidateExtractor | None = None,
        morph: pymorphy3.MorphAnalyzer | None = None,
    ) -> None:
        self._candidate_extractor = candidate_extractor or CandidateExtractor()
        self._morph = morph or pymorphy3.MorphAnalyzer()

    def extract(
        self,
        review: ReviewInput,
        category_id: str,
        vocabulary: Vocabulary,
    ) -> ResidualResult:
        candidates = self._candidate_extractor.extract(review.clean_text)
        relevant_aspects = self._select_relevant_aspects(
            category_id=category_id,
            vocabulary=vocabulary,
        )
        aspect_lemma_map = {
            aspect.id: self._build_aspect_lemma_set(aspect)
            for aspect in relevant_aspects
        }

        covered_phrases: list[str] = []
        covered_aspects: list[str] = []
        residual_phrases: list[str] = []
        seen_covered_aspects: set[str] = set()

        for candidate in candidates:
            phrase = candidate.span.strip()
            if not phrase:
                continue

            phrase_lemmas = self._lemmatize_text(phrase)
            matched_aspect_ids = [
                aspect_id
                for aspect_id, aspect_lemmas in aspect_lemma_map.items()
                if phrase_lemmas & aspect_lemmas
            ]

            if matched_aspect_ids:
                covered_phrases.append(phrase)
                for aspect_id in matched_aspect_ids:
                    if aspect_id in seen_covered_aspects:
                        continue
                    covered_aspects.append(aspect_id)
                    seen_covered_aspects.add(aspect_id)
            else:
                residual_phrases.append(phrase)

        return ResidualResult(
            review_id=review.id,
            covered_phrases=covered_phrases,
            covered_aspects=covered_aspects,
            residual_phrases=residual_phrases,
        )

    def _select_relevant_aspects(
        self,
        *,
        category_id: str,
        vocabulary: Vocabulary,
    ) -> list[AspectDefinition]:
        _ = category_id
        return list(vocabulary.aspects)

    def _build_aspect_lemma_set(self, aspect: AspectDefinition) -> set[str]:
        lemmas = self._lemmatize_text(aspect.canonical_name)
        for synonym in aspect.synonyms:
            lemmas.update(self._lemmatize_text(synonym))
        return lemmas

    def _lemmatize_text(self, text: str) -> set[str]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        return {self._lemmatize_token(token) for token in tokens if token}

    def _lemmatize_token(self, token: str) -> str:
        parses = self._morph.parse(token)
        if not parses:
            return token.lower()
        return str(parses[0].normal_form or token).lower()
