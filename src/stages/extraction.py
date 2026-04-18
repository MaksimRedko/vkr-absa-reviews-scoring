from __future__ import annotations

import re
from typing import List, Literal, Optional

import pymorphy3

from configs.configs import config
from src.schemas.models import Candidate
from src.stages.contracts import ExtractionStage
from src.stages.parsing import DependencyParser, ParsedChunk

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


def _clean_text(text: str) -> str:
    text = str(text).replace("\\n", " ")
    text = re.sub(r"(?<=[.,!?;:])(?=[^\s])", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _build_dependency_parser_from_config() -> DependencyParser:
    fallback_models = list(
        getattr(
            config.discovery,
            "dependency_spacy_fallback_models",
            ["ru_core_news_md", "ru_core_news_sm"],
        )
    )
    return DependencyParser(
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
            self._dependency_parser = _build_dependency_parser_from_config()

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
        return _clean_text(text)

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


class NounChunkExtractor(ExtractionStage):
    def __init__(self):
        self.last_filter_stats: dict[str, object] = {}
        self._dependency_parser = _build_dependency_parser_from_config()
        self._fallback_extractor = CandidateExtractor()
        self.allowed_chunk_dep_labels = {
            "nsubj",
            "obj",
            "ROOT",
            "appos",
            "conj",
        }

    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = _clean_text(raw_text)
        parsed = self._dependency_parser.parse(cleaned_text)
        if parsed.parse_failed or (not parsed.parser_available):
            fallback_candidates = self._fallback_extractor.extract(raw_text)
            self.last_filter_stats = {
                "extractor": "chunks",
                "parser_available": parsed.parser_available,
                "parse_failed": parsed.parse_failed,
                "fallback_used": True,
                "chunks_before_filter": 0,
                "chunks_after_filter": len(fallback_candidates),
            }
            return fallback_candidates

        candidates: list[Candidate] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        chunks_before_filter = len(parsed.noun_chunks)
        for chunk in parsed.noun_chunks:
            if chunk.dep_label not in self.allowed_chunk_dep_labels and not chunk.dep_label.startswith("nsubj"):
                continue
            key = (chunk.head_lemma, chunk.sentence, chunk.token_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                Candidate(
                    span=chunk.head_lemma,
                    sentence=chunk.sentence,
                    token_indices=chunk.token_indices,
                    source_span=chunk.text,
                    head_lemma=chunk.head_lemma,
                    modifier_text=chunk.modifier_text,
                    modifier_lemma=chunk.modifier_lemma,
                    modifier_type=chunk.modifier_type,
                    dep_label=chunk.dep_label,
                )
            )

        self.last_filter_stats = {
            "extractor": "chunks",
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
            "parser_model": parsed.model_name,
            "fallback_used": False,
            "chunks_before_filter": chunks_before_filter,
            "chunks_after_filter": len(candidates),
            "head_lemmas": sorted({candidate.head_lemma for candidate in candidates}),
        }
        return candidates


class AspectSentimentPairExtractor(ExtractionStage):
    def __init__(self, dependency_parser: DependencyParser | None = None):
        self.last_filter_stats: dict[str, object] = {}
        self._dependency_parser = dependency_parser or _build_dependency_parser_from_config()
        self._fallback_extractor = CandidateExtractor()

    @staticmethod
    def _is_informative_pair(pair: ParsedChunk) -> bool:
        return bool(pair.head_lemma) and pair.head_lemma not in STOP_TOKENS

    def extract_from_parsed(self, parsed) -> List[Candidate]:
        candidates: list[Candidate] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        for pair in parsed.aspect_pairs:
            if not self._is_informative_pair(pair):
                continue
            canonical_span = pair.canonical_span or pair.head_lemma
            surface_span = pair.pair_text or pair.text
            if not canonical_span or not surface_span:
                continue
            key = (canonical_span, pair.sentence, pair.token_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                Candidate(
                    span=canonical_span,
                    sentence=pair.sentence,
                    token_indices=pair.token_indices,
                    source_span=surface_span,
                    head_lemma=pair.head_lemma,
                    modifier_text=pair.modifier_text,
                    modifier_lemma=pair.modifier_lemma,
                    modifier_type=pair.modifier_type,
                    dep_label=pair.dep_label,
                )
            )

        return candidates

    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = _clean_text(raw_text)
        parsed = self._dependency_parser.parse(cleaned_text)
        if parsed.parse_failed or (not parsed.parser_available):
            fallback_candidates = self._fallback_extractor.extract(raw_text)
            self.last_filter_stats = {
                "extractor": "pairs",
                "parser_available": parsed.parser_available,
                "parse_failed": parsed.parse_failed,
                "fallback_used": True,
                "pairs_before_filter": 0,
                "pairs_after_filter": len(fallback_candidates),
            }
            return fallback_candidates

        candidates = self.extract_from_parsed(parsed)

        self.last_filter_stats = {
            "extractor": "pairs",
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
            "parser_model": parsed.model_name,
            "fallback_used": False,
            "pairs_before_filter": len(parsed.aspect_pairs),
            "pairs_after_filter": len(candidates),
            "unique_head_lemmas": sorted({candidate.head_lemma for candidate in candidates}),
            "modifier_types": sorted(
                {
                    str(candidate.modifier_type)
                    for candidate in candidates
                    if candidate.modifier_type is not None
                }
            ),
        }
        return candidates


class EventCandidateExtractor(ExtractionStage):
    def __init__(self, dependency_parser: DependencyParser | None = None):
        self.last_filter_stats: dict[str, object] = {}
        self._dependency_parser = dependency_parser or _build_dependency_parser_from_config()
        self._fallback_extractor = CandidateExtractor()

    def extract_from_parsed(self, parsed) -> List[Candidate]:
        candidates: list[Candidate] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        for event in parsed.event_candidates:
            canonical_span = event.canonical_span
            surface_span = event.pair_text or event.text
            if not canonical_span or not surface_span:
                continue
            key = (canonical_span, event.sentence, event.token_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                Candidate(
                    span=canonical_span,
                    sentence=event.sentence,
                    token_indices=event.token_indices,
                    source_span=surface_span,
                    head_lemma="",
                    modifier_text=event.modifier_text,
                    modifier_lemma=event.modifier_lemma,
                    modifier_type="event",
                    dep_label=event.dep_label,
                )
            )
        return candidates

    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = _clean_text(raw_text)
        parsed = self._dependency_parser.parse(cleaned_text)
        if parsed.parse_failed or (not parsed.parser_available):
            fallback_candidates = self._fallback_extractor.extract(raw_text)
            self.last_filter_stats = {
                "extractor": "events",
                "parser_available": parsed.parser_available,
                "parse_failed": parsed.parse_failed,
                "fallback_used": True,
                "events_before_filter": 0,
                "events_after_filter": len(fallback_candidates),
            }
            return fallback_candidates

        candidates = self.extract_from_parsed(parsed)
        self.last_filter_stats = {
            "extractor": "events",
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
            "parser_model": parsed.model_name,
            "fallback_used": False,
            "events_before_filter": len(parsed.event_candidates),
            "events_after_filter": len(candidates),
            "event_labels": sorted({candidate.span for candidate in candidates}),
        }
        return candidates


class NominalAspectExtractor(ExtractionStage):
    def __init__(self, dependency_parser: DependencyParser | None = None):
        self.last_filter_stats: dict[str, object] = {}
        self._dependency_parser = dependency_parser or _build_dependency_parser_from_config()
        self._fallback_extractor = CandidateExtractor()

    def extract_from_parsed(self, parsed) -> List[Candidate]:
        candidates: list[Candidate] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        for nominal in parsed.nominal_candidates:
            canonical_span = nominal.canonical_span or nominal.head_lemma
            surface_span = nominal.pair_text or nominal.text
            if not canonical_span or not surface_span:
                continue
            key = (canonical_span, nominal.sentence, nominal.token_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                Candidate(
                    span=canonical_span,
                    sentence=nominal.sentence,
                    token_indices=nominal.token_indices,
                    source_span=surface_span,
                    head_lemma=nominal.head_lemma,
                    modifier_text="",
                    modifier_lemma="",
                    modifier_type="nominal",
                    dep_label=nominal.dep_label,
                )
            )
        return candidates

    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = _clean_text(raw_text)
        parsed = self._dependency_parser.parse(cleaned_text)
        if parsed.parse_failed or (not parsed.parser_available):
            fallback_candidates = self._fallback_extractor.extract(raw_text)
            self.last_filter_stats = {
                "extractor": "nominals",
                "parser_available": parsed.parser_available,
                "parse_failed": parsed.parse_failed,
                "fallback_used": True,
                "nominals_before_filter": 0,
                "nominals_after_filter": len(fallback_candidates),
            }
            return fallback_candidates

        candidates = self.extract_from_parsed(parsed)
        self.last_filter_stats = {
            "extractor": "nominals",
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
            "parser_model": parsed.model_name,
            "fallback_used": False,
            "nominals_before_filter": len(parsed.nominal_candidates),
            "nominals_after_filter": len(candidates),
            "head_lemmas": sorted({candidate.head_lemma for candidate in candidates}),
        }
        return candidates


class PairAndEventExtractor(ExtractionStage):
    def __init__(self):
        self.last_filter_stats: dict[str, object] = {}
        shared_parser = _build_dependency_parser_from_config()
        self._dependency_parser = shared_parser
        self._fallback_extractor = CandidateExtractor()
        self._pair_extractor = AspectSentimentPairExtractor(dependency_parser=shared_parser)
        self._event_extractor = EventCandidateExtractor(dependency_parser=shared_parser)

    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = _clean_text(raw_text)
        parsed = self._dependency_parser.parse(cleaned_text)
        if parsed.parse_failed or (not parsed.parser_available):
            fallback_candidates = self._fallback_extractor.extract(raw_text)
            self.last_filter_stats = {
                "extractor": "pairs+events",
                "parser_available": parsed.parser_available,
                "parse_failed": parsed.parse_failed,
                "fallback_used": True,
                "pairs_before_filter": 0,
                "events_before_filter": 0,
                "candidates_after_filter": len(fallback_candidates),
            }
            return fallback_candidates

        candidates: list[Candidate] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        pair_candidates = self._pair_extractor.extract_from_parsed(parsed)
        event_candidates = self._event_extractor.extract_from_parsed(parsed)
        for candidate in (
            pair_candidates
            + event_candidates
        ):
            key = (candidate.span, candidate.sentence, candidate.token_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)

        self.last_filter_stats = {
            "extractor": "pairs+events",
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
            "parser_model": parsed.model_name,
            "fallback_used": False,
            "pairs_before_filter": len(parsed.aspect_pairs),
            "events_before_filter": len(parsed.event_candidates),
            "pairs_after_filter": len(pair_candidates),
            "events_after_filter": len(event_candidates),
            "candidates_after_filter": len(candidates),
        }
        return candidates


class PairEventNominalExtractor(ExtractionStage):
    def __init__(self):
        self.last_filter_stats: dict[str, object] = {}
        shared_parser = _build_dependency_parser_from_config()
        self._dependency_parser = shared_parser
        self._fallback_extractor = CandidateExtractor()
        self._pair_extractor = AspectSentimentPairExtractor(dependency_parser=shared_parser)
        self._event_extractor = EventCandidateExtractor(dependency_parser=shared_parser)
        self._nominal_extractor = NominalAspectExtractor(dependency_parser=shared_parser)

    def extract(self, raw_text: str) -> List[Candidate]:
        cleaned_text = _clean_text(raw_text)
        parsed = self._dependency_parser.parse(cleaned_text)
        if parsed.parse_failed or (not parsed.parser_available):
            fallback_candidates = self._fallback_extractor.extract(raw_text)
            self.last_filter_stats = {
                "extractor": "pairs+events+nominals",
                "parser_available": parsed.parser_available,
                "parse_failed": parsed.parse_failed,
                "fallback_used": True,
                "pairs_before_filter": 0,
                "events_before_filter": 0,
                "nominals_before_filter": 0,
                "candidates_after_filter": len(fallback_candidates),
            }
            return fallback_candidates

        candidates: list[Candidate] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        pair_candidates = self._pair_extractor.extract_from_parsed(parsed)
        event_candidates = self._event_extractor.extract_from_parsed(parsed)
        nominal_candidates = self._nominal_extractor.extract_from_parsed(parsed)
        for candidate in (
            pair_candidates
            + event_candidates
            + nominal_candidates
        ):
            key = (candidate.span, candidate.sentence, candidate.token_indices)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)

        self.last_filter_stats = {
            "extractor": "pairs+events+nominals",
            "parser_available": parsed.parser_available,
            "parse_failed": parsed.parse_failed,
            "parser_model": parsed.model_name,
            "fallback_used": False,
            "pairs_before_filter": len(parsed.aspect_pairs),
            "events_before_filter": len(parsed.event_candidates),
            "nominals_before_filter": len(parsed.nominal_candidates),
            "pairs_after_filter": len(pair_candidates),
            "events_after_filter": len(event_candidates),
            "nominals_after_filter": len(nominal_candidates),
            "candidates_after_filter": len(candidates),
        }
        return candidates


def build_extraction_stage() -> ExtractionStage:
    extractor_name = str(getattr(config.discovery, "extractor", "ngram"))
    if extractor_name == "pairs+events+nominals":
        return PairEventNominalExtractor()
    if extractor_name == "pairs+events":
        return PairAndEventExtractor()
    if extractor_name == "nominals":
        return NominalAspectExtractor()
    if extractor_name == "events":
        return EventCandidateExtractor()
    if extractor_name == "pairs":
        return AspectSentimentPairExtractor()
    if extractor_name == "chunks":
        return NounChunkExtractor()
    return CandidateExtractor()


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
