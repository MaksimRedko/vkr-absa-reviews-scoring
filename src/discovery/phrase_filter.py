from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import pymorphy3

_TOKEN_RE = re.compile(r"[\w\-₽.]+", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d", flags=re.UNICODE)
_NUMERAL_WORDS = {
    "первый",
    "второй",
    "третий",
    "четвертый",
    "четвёртый",
    "пятый",
    "шестой",
    "седьмой",
    "восьмой",
    "девятый",
    "десятый",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
}
_MONTH_LEMMAS = {
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
}
_MONEY_TOKENS = {
    "руб",
    "р",
    "р.",
    "₽",
    "рублей",
    "рубль",
    "рубля",
    "доллар",
    "евро",
    "коп",
    "р/мес",
}
_CONTEXT_STOP_LEMMAS = {
    "товар",
    "вещь",
    "заказ",
    "покупка",
    "продавец",
    "магазин",
    "покупатель",
    "доставщик",
    "отправитель",
    "получатель",
    "артикул",
    "каталог",
    "бренд",
    "производитель",
    "штука",
    "экземпляр",
}
_RULES = ("numeric", "monetary", "temporal", "service_only", "too_short", "context_stop")


@dataclass(slots=True)
class FilterReport:
    total_input: int
    total_kept: int
    total_filtered: int
    filtered_by_rule: dict[str, int]
    filter_rate: float
    sample_filtered: dict[str, list[str]]


class PhraseFilter:
    def __init__(self, *, morph: pymorphy3.MorphAnalyzer | None = None) -> None:
        self._morph = morph or pymorphy3.MorphAnalyzer()

    def filter(self, phrases: list[str]) -> tuple[list[str], FilterReport]:
        kept: list[str] = []
        counts: Counter[str] = Counter()
        samples: dict[str, list[str]] = {rule: [] for rule in _RULES}

        for phrase in phrases:
            clean = str(phrase).strip()
            reason = self._filter_reason(clean)
            if reason is None:
                kept.append(clean)
                continue
            counts[reason] += 1
            if len(samples[reason]) < 5:
                samples[reason].append(clean)

        total = len(phrases)
        filtered = total - len(kept)
        report = FilterReport(
            total_input=total,
            total_kept=len(kept),
            total_filtered=filtered,
            filtered_by_rule={rule: int(counts.get(rule, 0)) for rule in _RULES},
            filter_rate=float(filtered) / float(total) if total else 0.0,
            sample_filtered=samples,
        )
        return kept, report

    def _filter_reason(self, phrase: str) -> str | None:
        if not phrase:
            return "too_short"

        tokens = self._tokens(phrase)
        if self._is_numeric_phrase(phrase, tokens):
            return "numeric"
        if self._is_monetary_phrase(tokens):
            return "monetary"

        analyses = [self._analyze(token) for token in tokens]
        lemmas = [lemma for lemma, _pos in analyses]

        if self._is_temporal_phrase(tokens, lemmas):
            return "temporal"
        if self._is_service_only(analyses):
            return "service_only"
        if self._is_too_short(phrase, analyses):
            return "too_short"
        if self._is_context_stop(lemmas):
            return "context_stop"
        return None

    def _tokens(self, phrase: str) -> list[str]:
        return [
            token.lower().strip()
            for token in _TOKEN_RE.findall(phrase)
            if token.strip() and not all(char == "." for char in token.strip())
        ]

    def _analyze(self, token: str) -> tuple[str, str]:
        parses = self._morph.parse(token)
        if not parses:
            return token.lower(), ""
        parse = parses[0]
        return str(parse.normal_form or token).lower(), str(parse.tag.POS or "")

    def _is_numeric_phrase(self, phrase: str, tokens: list[str]) -> bool:
        if not tokens:
            return False
        digit_count = len(_DIGIT_RE.findall(phrase))
        non_space_count = sum(1 for char in phrase if not char.isspace())
        if non_space_count and digit_count / non_space_count >= 0.5:
            return True
        return all(any(char.isdigit() for char in token) for token in tokens)

    def _is_monetary_phrase(self, tokens: list[str]) -> bool:
        normalized = {token.rstrip(".") for token in tokens}
        if "₽" in "".join(tokens):
            return True
        return bool(normalized & {token.rstrip(".") for token in _MONEY_TOKENS})

    def _is_temporal_phrase(self, tokens: list[str], lemmas: list[str]) -> bool:
        if any(lemma in _MONTH_LEMMAS for lemma in lemmas):
            return True
        has_number = any(any(char.isdigit() for char in token) for token in tokens)
        if has_number and any(lemma in {"год", "лето"} or lemma == "год" for lemma in lemmas):
            return True
        if "раз" in lemmas:
            has_numeral = has_number or any(lemma in _NUMERAL_WORDS for lemma in lemmas)
            if has_numeral or len(tokens) == 1:
                return True
        if len(tokens) == 1 and lemmas[0] in {"месяц", "год", "день", "неделя"}:
            return True
        return False

    def _is_service_only(self, analyses: list[tuple[str, str]]) -> bool:
        if not analyses:
            return False
        return all(pos != "NOUN" for _lemma, pos in analyses)

    def _is_too_short(self, phrase: str, analyses: list[tuple[str, str]]) -> bool:
        if len(phrase.strip()) >= 4:
            return False
        return not any(pos == "NOUN" for _lemma, pos in analyses)

    def _is_context_stop(self, lemmas: list[str]) -> bool:
        return bool(lemmas) and all(lemma in _CONTEXT_STOP_LEMMAS for lemma in lemmas)
