from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable, Literal, Optional

import pymorphy3

from src.stages.lexicons import NEUTRAL_ADVERBS, STOP_EVENT_LEMMAS, STOP_NOMINAL_LEMMAS

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None


_morph = pymorphy3.MorphAnalyzer()
_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = Lock()


@dataclass
class ParsedDocument:
    head_lemmas: set[str]
    aspect_role_lemmas: set[str]
    model_name: Optional[str]
    parser_available: bool
    parse_failed: bool = False
    noun_chunks: list["ParsedChunk"] = field(default_factory=list)
    aspect_pairs: list["ParsedChunk"] = field(default_factory=list)
    event_candidates: list["ParsedChunk"] = field(default_factory=list)
    nominal_candidates: list["ParsedChunk"] = field(default_factory=list)


@dataclass
class ParsedChunk:
    text: str
    head_lemma: str
    sentence: str
    token_indices: tuple[int, int]
    dep_label: str
    modifier_text: str = ""
    modifier_lemma: str = ""
    modifier_type: Optional[Literal["amod", "xcomp", "copular", "predicative", "event", "nominal"]] = None
    pair_text: str = ""
    canonical_span: str = ""


class DependencyParser:
    _CHUNK_CHILD_DEPS = {"amod", "det", "nummod", "compound", "nmod", "flat", "fixed"}
    _ASPECT_PHRASE_CHILD_DEPS = {"det", "nummod", "compound", "flat", "fixed"}
    _PAIR_ASPECT_DEPS = {"nsubj", "nsubj:pass", "obj", "iobj", "ROOT", "appos", "conj"}
    _ATTRIBUTIVE_MODIFIER_DEPS = {"amod", "acl", "xcomp"}
    _PREDICATIVE_POS = {"ADJ", "ADV", "VERB"}
    _MODIFIER_CHILD_DEPS = {"advmod", "aux", "cop", "fixed", "flat", "compound", "mark", "det", "nummod"}
    _EVENT_DEPS = {"ROOT", "acl", "advcl", "ccomp", "xcomp"}
    _EVENT_POS = {"ADJ", "ADV", "VERB"}
    _EVENT_EXCLUDED_POS = {"PUNCT", "SCONJ", "CCONJ"}
    _EVENT_SIGNAL_POS = {"ADJ", "ADV"}
    _NOMINAL_DEPS = {"nsubj", "nsubj:pass", "obj", "ROOT", "appos", "conj"}

    def __init__(
        self,
        preferred_model: str,
        fallback_models: Optional[Iterable[str]] = None,
        include_root_verbs: bool = True,
        include_root_adjs: bool = True,
    ):
        self.preferred_model = preferred_model
        self.fallback_models = list(fallback_models or [])
        self.include_root_verbs = bool(include_root_verbs)
        self.include_root_adjs = bool(include_root_adjs)
        self._loaded_model_name: Optional[str] = None
        self._nlp = None

    @staticmethod
    def _normalize_lemma(lemma: str) -> str:
        return (lemma or "").strip().lower()

    @staticmethod
    def _normalize_guard_lemma(lemma: str) -> str:
        normalized = (lemma or "").strip().lower()
        if not normalized:
            return ""
        parses = _morph.parse(normalized)
        if not parses:
            return normalized
        return str(parses[0].normal_form or normalized).lower()

    @staticmethod
    def _is_content_token(token) -> bool:
        return not (token.is_space or token.is_punct)

    @staticmethod
    def _is_head_noun(token) -> bool:
        if token.pos_ != "NOUN":
            return False
        if token.head is token:
            return True
        if token.head.pos_ != "NOUN":
            return True
        return token.dep_ not in {"nmod", "compound", "flat", "fixed", "appos"}

    @staticmethod
    def _is_aspect_role_token(
        token,
        include_root_verbs: bool,
        include_root_adjs: bool,
    ) -> bool:
        dep = str(token.dep_ or "")
        pos = str(token.pos_ or "")

        if pos in {"NOUN", "PROPN"} and (
            dep in {"nsubj", "nsubj:pass", "obj", "ROOT", "appos", "conj"}
            or dep.startswith("nsubj")
        ):
            return True

        if include_root_verbs and pos == "VERB" and dep in {"ROOT", "conj"}:
            return True

        # spaCy often tags predicatives like "дорого" as ADV rather than ADJ.
        if include_root_adjs and pos in {"ADJ", "ADV"} and dep in {"ROOT", "conj"}:
            return True

        return False

    def _candidate_model_names(self) -> list[str]:
        names = [self.preferred_model, *self.fallback_models]
        ordered: list[str] = []
        seen: set[str] = set()
        for name in names:
            normalized = str(name or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    @staticmethod
    def _is_chunk_root(token) -> bool:
        dep = str(token.dep_ or "")
        pos = str(token.pos_ or "")
        if pos not in {"NOUN", "PROPN"}:
            return False
        return dep in {"nsubj", "obj", "ROOT", "appos", "conj"} or dep.startswith("nsubj")

    @classmethod
    def _is_pair_aspect_token(cls, token) -> bool:
        dep = str(token.dep_ or "")
        pos = str(token.pos_ or "")
        if pos not in {"NOUN", "PROPN"}:
            return False
        return dep in cls._PAIR_ASPECT_DEPS or dep.startswith("nsubj")

    @classmethod
    def _is_event_token(cls, token) -> bool:
        dep = str(token.dep_ or "")
        pos = str(token.pos_ or "")
        return pos in cls._EVENT_POS and dep in cls._EVENT_DEPS

    @staticmethod
    def _render_tokens(tokens: list) -> str:
        return " ".join(
            str(token.text).lower()
            for token in tokens
            if not (token.is_space or token.is_punct)
        ).strip()

    @staticmethod
    def _relative_token_indices(tokens: list, sentence_tokens: list) -> tuple[int, int]:
        sentence_start = sentence_tokens[0].i
        return (
            int(min(token.i for token in tokens) - sentence_start),
            int(max(token.i for token in tokens) - sentence_start + 1),
        )

    @staticmethod
    def _collect_tokens_by_deps(root_token, allowed_deps: set[str]) -> list:
        tokens = {root_token}
        stack = [root_token]
        while stack:
            current = stack.pop()
            for child in current.children:
                if child.dep_ in allowed_deps:
                    if child not in tokens:
                        tokens.add(child)
                        stack.append(child)
        return sorted(tokens, key=lambda token: token.i)

    def _collect_chunk_tokens(self, root_token) -> list:
        return self._collect_tokens_by_deps(root_token, self._CHUNK_CHILD_DEPS)

    def _collect_aspect_phrase_tokens(self, root_token) -> list:
        return self._collect_tokens_by_deps(root_token, self._ASPECT_PHRASE_CHILD_DEPS)

    def _collect_modifier_tokens(self, root_token, excluded_ids: set[int]) -> list:
        tokens = set()
        stack = [root_token]
        while stack:
            current = stack.pop()
            if current.i in excluded_ids:
                continue
            if current.is_space or current.is_punct:
                continue
            if current not in tokens:
                tokens.add(current)
            for child in current.children:
                if child.i in excluded_ids:
                    continue
                if child.dep_ in self._MODIFIER_CHILD_DEPS:
                    stack.append(child)
        return sorted(tokens, key=lambda token: token.i)

    def _collect_event_tokens(self, root_token) -> list:
        return [
            token
            for token in sorted(root_token.subtree, key=lambda item: item.i)
            if (
                not token.is_space
                and not token.is_punct
                and str(token.pos_ or "") not in self._EVENT_EXCLUDED_POS
            )
        ]

    def _build_chunk(self, root_token) -> Optional["ParsedChunk"]:
        chunk_tokens = self._collect_chunk_tokens(root_token)
        if not chunk_tokens:
            return None
        surface_tokens = [token for token in chunk_tokens if not (token.is_space or token.is_punct)]
        if not surface_tokens:
            return None
        head_lemma = self._normalize_lemma(root_token.lemma_)
        if not head_lemma:
            return None
        sentence_tokens = [
            token
            for token in root_token.sent
            if not (token.is_space or token.is_punct)
        ]
        if not sentence_tokens:
            return None
        return ParsedChunk(
            text=self._render_tokens(surface_tokens),
            head_lemma=head_lemma,
            sentence=str(root_token.sent.text).strip(),
            token_indices=self._relative_token_indices(surface_tokens, sentence_tokens),
            dep_label=str(root_token.dep_ or ""),
        )

    def _find_predicative_modifier(
        self,
        aspect_token,
    ) -> tuple[object, Literal["copular", "predicative"]] | None:
        head = aspect_token.head
        if head is None or head is aspect_token:
            return None
        if str(head.pos_ or "") not in self._PREDICATIVE_POS:
            return None
        modifier_type: Literal["copular", "predicative"] = "predicative"
        if any(str(child.dep_ or "") == "cop" for child in head.children):
            modifier_type = "copular"
        return head, modifier_type

    def _build_pair_chunk(
        self,
        aspect_token,
        modifier_token=None,
        modifier_type: Optional[Literal["amod", "xcomp", "copular", "predicative"]] = None,
    ) -> Optional["ParsedChunk"]:
        aspect_tokens = [
            token
            for token in self._collect_aspect_phrase_tokens(aspect_token)
            if not (token.is_space or token.is_punct)
        ]
        if not aspect_tokens:
            return None
        sentence_tokens = [
            token
            for token in aspect_token.sent
            if not (token.is_space or token.is_punct)
        ]
        if not sentence_tokens:
            return None
        head_lemma = self._normalize_lemma(aspect_token.lemma_)
        if not head_lemma:
            return None

        modifier_text = ""
        modifier_lemma = ""
        pair_tokens = list(aspect_tokens)
        if modifier_token is not None:
            modifier_lemma = self._normalize_lemma(modifier_token.lemma_)
            if not modifier_lemma:
                return None
            modifier_tokens = self._collect_modifier_tokens(
                modifier_token,
                excluded_ids={token.i for token in aspect_tokens},
            )
            if not modifier_tokens:
                return None
            modifier_text = self._render_tokens(modifier_tokens)
            if not modifier_text:
                return None
            pair_tokens = sorted(
                {token for token in (*aspect_tokens, *modifier_tokens)},
                key=lambda token: token.i,
            )

        pair_text = self._render_tokens(pair_tokens)
        if not pair_text:
            return None
        canonical_span = head_lemma if not modifier_lemma else f"{head_lemma}_{modifier_lemma}"
        return ParsedChunk(
            text=self._render_tokens(aspect_tokens),
            head_lemma=head_lemma,
            sentence=str(aspect_token.sent.text).strip(),
            token_indices=self._relative_token_indices(pair_tokens, sentence_tokens),
            dep_label=str(aspect_token.dep_ or ""),
            modifier_text=modifier_text,
            modifier_lemma=modifier_lemma,
            modifier_type=modifier_type,
            pair_text=pair_text,
            canonical_span=canonical_span,
        )

    @classmethod
    def _is_nominal_candidate_token(cls, token) -> bool:
        dep = str(token.dep_ or "")
        pos = str(token.pos_ or "")
        return pos in {"NOUN", "PROPN"} and (
            dep in cls._NOMINAL_DEPS or dep.startswith("nsubj")
        )

    def _build_nominal_candidate(self, aspect_token) -> Optional["ParsedChunk"]:
        head_lemma = self._normalize_guard_lemma(aspect_token.lemma_)
        if not head_lemma or head_lemma in STOP_NOMINAL_LEMMAS:
            return None
        aspect_tokens = [
            token
            for token in self._collect_aspect_phrase_tokens(aspect_token)
            if not (token.is_space or token.is_punct)
        ]
        if not aspect_tokens:
            return None
        sentence_tokens = [
            token
            for token in aspect_token.sent
            if not (token.is_space or token.is_punct)
        ]
        if not sentence_tokens:
            return None
        aspect_text = self._render_tokens(aspect_tokens)
        if not aspect_text:
            return None
        return ParsedChunk(
            text=aspect_text,
            head_lemma=head_lemma,
            sentence=str(aspect_token.sent.text).strip(),
            token_indices=self._relative_token_indices(aspect_tokens, sentence_tokens),
            dep_label=str(aspect_token.dep_ or ""),
            modifier_type="nominal",
            pair_text=aspect_text,
            canonical_span=head_lemma,
        )

    @staticmethod
    def _has_subject_noun(event_token) -> bool:
        for token in event_token.subtree:
            if str(token.pos_ or "") not in {"NOUN", "PROPN"}:
                continue
            dep = str(token.dep_ or "")
            if dep.startswith("nsubj"):
                return True
        return False

    @classmethod
    def _has_event_signal(cls, event_token, event_tokens: list) -> bool:
        event_pos = str(event_token.pos_ or "")
        if event_pos in {"ADJ", "ADV"}:
            event_lemma = cls._normalize_guard_lemma(event_token.lemma_)
            return bool(event_lemma) and event_lemma not in NEUTRAL_ADVERBS
        for token in event_tokens:
            if token is event_token:
                continue
            token_pos = str(token.pos_ or "")
            token_lemma = cls._normalize_guard_lemma(token.lemma_)
            if (
                token_pos in cls._EVENT_SIGNAL_POS
                and token_lemma
                and token_lemma not in NEUTRAL_ADVERBS
            ):
                return True
            if str(token.dep_ or "") == "neg":
                return True
        return False

    @classmethod
    def _has_stop_copula(cls, event_token) -> bool:
        for child in event_token.children:
            if str(child.dep_ or "") not in {"cop", "aux"}:
                continue
            if cls._normalize_guard_lemma(child.lemma_) in STOP_EVENT_LEMMAS:
                return True
        return False

    def _build_event_candidate(self, event_token) -> Optional["ParsedChunk"]:
        lemma = self._normalize_guard_lemma(event_token.lemma_)
        if not lemma or lemma in STOP_EVENT_LEMMAS:
            return None
        if str(event_token.pos_ or "") == "VERB" and self._has_stop_copula(event_token):
            return None
        if self._has_subject_noun(event_token):
            return None

        event_tokens = self._collect_event_tokens(event_token)
        if not self._has_event_signal(event_token, event_tokens):
            return None

        sentence_tokens = [
            token
            for token in event_token.sent
            if not (token.is_space or token.is_punct)
        ]
        if not sentence_tokens:
            return None

        source_span = self._render_tokens(event_tokens)
        if not source_span:
            return None

        return ParsedChunk(
            text=source_span,
            head_lemma="",
            sentence=str(event_token.sent.text).strip(),
            token_indices=self._relative_token_indices(event_tokens, sentence_tokens),
            dep_label=str(event_token.dep_ or ""),
            modifier_text=source_span,
            modifier_lemma=lemma,
            modifier_type="event",
            pair_text=source_span,
            canonical_span=f"event_{lemma}",
        )

    def _build_aspect_pairs(self, aspect_token) -> list["ParsedChunk"]:
        pairs: list[ParsedChunk] = []
        seen: set[tuple[str, str]] = set()

        for child in aspect_token.children:
            child_dep = str(child.dep_ or "")
            child_pos = str(child.pos_ or "")
            if child_dep not in self._ATTRIBUTIVE_MODIFIER_DEPS:
                continue
            if child_pos not in {"ADJ", "VERB", "ADV"}:
                continue
            modifier_type: Literal["amod", "xcomp", "copular", "predicative"]
            modifier_type = "amod" if child_dep == "amod" else "xcomp"
            pair = self._build_pair_chunk(aspect_token, child, modifier_type)
            if pair is None:
                continue
            key = (pair.canonical_span, pair.pair_text)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)

        if str(aspect_token.dep_ or "").startswith("nsubj"):
            predicative = self._find_predicative_modifier(aspect_token)
            if predicative is not None:
                modifier_token, modifier_type = predicative
                pair = self._build_pair_chunk(aspect_token, modifier_token, modifier_type)
                if pair is not None:
                    key = (pair.canonical_span, pair.pair_text)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(pair)

        return pairs

    def _load_model(self) -> tuple[object | None, Optional[str]]:
        if spacy is None:
            return None, None
        if self._nlp is not None:
            return self._nlp, self._loaded_model_name

        for model_name in self._candidate_model_names():
            with _MODEL_CACHE_LOCK:
                if model_name in _MODEL_CACHE:
                    self._nlp = _MODEL_CACHE[model_name]
                    self._loaded_model_name = model_name
                    return self._nlp, model_name
            try:
                nlp = spacy.load(model_name, disable=["ner"])
            except (IOError, OSError, ValueError):
                continue
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[model_name] = nlp
            self._nlp = nlp
            self._loaded_model_name = model_name
            return nlp, model_name
        return None, None

    def parse(self, text: str) -> ParsedDocument:
        text = str(text or "").strip()
        if not text:
            return ParsedDocument(
                head_lemmas=set(),
                aspect_role_lemmas=set(),
                model_name=None,
                parser_available=False,
                parse_failed=False,
                noun_chunks=[],
                aspect_pairs=[],
                event_candidates=[],
                nominal_candidates=[],
            )

        nlp, model_name = self._load_model()
        if nlp is None:
            return ParsedDocument(
                head_lemmas=set(),
                aspect_role_lemmas=set(),
                model_name=None,
                parser_available=False,
                parse_failed=True,
                noun_chunks=[],
                aspect_pairs=[],
                event_candidates=[],
                nominal_candidates=[],
            )

        try:
            doc = nlp(text)
        except (TypeError, ValueError, RuntimeError):
            return ParsedDocument(
                head_lemmas=set(),
                aspect_role_lemmas=set(),
                model_name=model_name,
                parser_available=True,
                parse_failed=True,
                noun_chunks=[],
                aspect_pairs=[],
                event_candidates=[],
                nominal_candidates=[],
            )

        head_lemmas: set[str] = set()
        aspect_role_lemmas: set[str] = set()
        noun_chunks: list[ParsedChunk] = []
        aspect_pairs: list[ParsedChunk] = []
        event_candidates: list[ParsedChunk] = []
        nominal_candidates: list[ParsedChunk] = []
        for token in doc:
            if not self._is_content_token(token):
                continue
            lemma = self._normalize_lemma(token.lemma_)
            if not lemma:
                continue
            if self._is_head_noun(token):
                head_lemmas.add(lemma)
            if self.include_root_verbs and token.pos_ == "VERB" and token.dep_ in {"ROOT", "conj"}:
                head_lemmas.add(lemma)
            if self.include_root_adjs and token.pos_ == "ADJ" and token.dep_ in {"ROOT", "conj"}:
                head_lemmas.add(lemma)

            if self._is_aspect_role_token(
                token,
                include_root_verbs=self.include_root_verbs,
                include_root_adjs=self.include_root_adjs,
            ):
                aspect_role_lemmas.add(lemma)
            if self._is_chunk_root(token):
                chunk = self._build_chunk(token)
                if chunk is not None:
                    noun_chunks.append(chunk)
            is_pair_token = self._is_pair_aspect_token(token)
            if is_pair_token:
                pairs = self._build_aspect_pairs(token)
                aspect_pairs.extend(pairs)
                if (not pairs) and self._is_nominal_candidate_token(token):
                    nominal_candidate = self._build_nominal_candidate(token)
                    if nominal_candidate is not None:
                        nominal_candidates.append(nominal_candidate)
            is_event_token = self._is_event_token(token)
            if is_event_token:
                event_candidate = self._build_event_candidate(token)
                if event_candidate is not None:
                    event_candidates.append(event_candidate)
            elif (not is_pair_token) and self._is_nominal_candidate_token(token):
                nominal_candidate = self._build_nominal_candidate(token)
                if nominal_candidate is not None:
                    nominal_candidates.append(nominal_candidate)

        return ParsedDocument(
            head_lemmas=head_lemmas,
            aspect_role_lemmas=aspect_role_lemmas,
            model_name=model_name,
            parser_available=True,
            parse_failed=False,
            noun_chunks=noun_chunks,
            aspect_pairs=aspect_pairs,
            event_candidates=event_candidates,
            nominal_candidates=nominal_candidates,
        )
