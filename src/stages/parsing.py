from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Optional

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None


_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = Lock()


@dataclass
class ParsedDocument:
    head_lemmas: set[str]
    aspect_role_lemmas: set[str]
    model_name: Optional[str]
    parser_available: bool
    parse_failed: bool = False


class DependencyParser:
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
            )

        nlp, model_name = self._load_model()
        if nlp is None:
            return ParsedDocument(
                head_lemmas=set(),
                aspect_role_lemmas=set(),
                model_name=None,
                parser_available=False,
                parse_failed=True,
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
            )

        head_lemmas: set[str] = set()
        aspect_role_lemmas: set[str] = set()
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

        return ParsedDocument(
            head_lemmas=head_lemmas,
            aspect_role_lemmas=aspect_role_lemmas,
            model_name=model_name,
            parser_available=True,
            parse_failed=False,
        )
