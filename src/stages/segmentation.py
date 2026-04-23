"""
RuleBasedClauseSegmenter v1.0

Discourse-aware clause segmentation for Russian ABSA.

This is NOT atomic claim decomposition. This is a rule-based, linear-time
clause segmenter that produces minimal evaluative/event-level units suitable
as units of analysis for downstream aspect/polarity scoring.

Pipeline:
  1. Normalization
  2. Hard split (terminal punctuation, with abbreviation/decimal protection)
  3. Soft split (contrast + opener discourse markers; conservative "а" handling)
  4. Orphan merge (short segments fold into the previous segment)

Reportative complementizers ("сказали, что ...") are explicitly NOT handled
in v1.0 — they are reserved for v1.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pymorphy3

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

CONTRAST_MARKERS: tuple[str, ...] = (
    "с одной стороны",
    "с другой стороны",
    "при этом",
    "в то же время",
    "однако",
    "зато",
    "хотя",
    "но",
)

OPENER_MARKERS: tuple[str, ...] = (
    "в общем",
    "в итоге",
    "кроме того",
    "поэтому",
    "также",
    "кстати",
)

A_NON_SPLIT_PREFIXES: tuple[str, ...] = (
    "а ещё и",
    "а ещё",
    "а также",
    "а вообще",
    "а потом",
    "а затем",
    "а после",
    "а вот",
    "а именно",
    "а то",
)

ABBREVIATIONS: tuple[str, ...] = (
    "т.е.",
    "и т.д.",
    "и т.п.",
    "т.д.",
    "т.п.",
    "т.к.",
    "т.н.",
    "др.",
    "пр.",
    "см.",
    "напр.",
    "г.",
    "ул.",
    "стр.",
)

SECTION_LABEL_MARKERS: tuple[str, ...] = (
    "достоинства:",
    "недостатки:",
    "преимущества:",
    "комментарий:",
)

TRIVIAL_TAIL_MARKERS: tuple[str, ...] = (
    "но",
    "и",
    "а",
    "или",
    "однако",
    "зато",
    "хотя",
    "при этом",
    "в то же время",
    "с одной стороны",
    "с другой стороны",
)

RUSSIAN_STOPLIST_MIN: frozenset[str] = frozenset({
    "это", "этот", "эта", "эти", "этого",
    "тот", "та", "те", "того",
    "был", "была", "было", "были", "быть",
    "его", "её", "их", "наш", "ваш", "мой", "твой",
    "что", "как", "так", "там", "тут", "где", "когда",
    "уже", "ещё", "тоже", "только", "очень",
    "для", "при", "над", "под", "без",
    "или", "либо", "ни", "не",
})

# Tokens from discourse markers / "а" whitelist that should not count as content.
_MARKER_TOKENS: frozenset[str] = frozenset(
    tok
    for phrase in (*CONTRAST_MARKERS, *OPENER_MARKERS, *A_NON_SPLIT_PREFIXES)
    for tok in phrase.split()
) | frozenset({"а"})

CONTENT_STOPLIST: frozenset[str] = RUSSIAN_STOPLIST_MIN | _MARKER_TOKENS

PREDICATIVE_POS: frozenset[str] = frozenset({"VERB", "INFN", "PRTF", "PRTS", "PRED", "ADJS"})

# Minimum content tokens for a segment to survive standalone (orphan merge threshold).
MIN_CONTENT_TOKENS: int = 3

# Minimum content tokens on each side of a discourse marker for split to fire.
# Lower than MIN_CONTENT_TOKENS because contrastive ABSA fragments can be legitimately
# short ("чисто, но шумно"). Contrast markers get an additional relaxed 1/1 guard below.
MIN_SPLIT_CONTENT: int = 2

# ---------------------------------------------------------------------------
# Morphology (singleton + cache)
# ---------------------------------------------------------------------------

_MORPH = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _is_predicative(token: str) -> bool:
    """True if token's best parse has a predicative POS tag."""
    parses = _MORPH.parse(token)
    if not parses:
        return False
    pos = parses[0].tag.POS
    return pos in PREDICATIVE_POS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    segment_index: int
    char_start: int
    char_end: int
    source_text: str
    boundary_type: frozenset[str]
    source_review_id: str | None = None


@dataclass(slots=True)
class _Chunk:
    """Internal mutable chunk used during pipeline stages."""
    text: str
    char_start: int
    char_end: int
    boundary_type: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

class RuleBasedClauseSegmenter:
    """Linear-time discourse-aware clause segmenter for Russian reviews."""

    # Terminal punctuation that triggers hard boundaries.
    _TERMINAL_CHARS: frozenset[str] = frozenset(".!?;")

    _RE_NORMALIZE_ELLIPSIS = re.compile(r"(?:\.{2,}|…+)")
    _RE_NORMALIZE_MULTI_BANG = re.compile(r"!{2,}")
    _RE_NORMALIZE_MULTI_QMARK = re.compile(r"\?{2,}")
    _RE_NORMALIZE_INTERROBANG = re.compile(r"[!?]*\?[!?]*")  # any mix containing ?
    _RE_NORMALIZE_BANGSEQ = re.compile(r"!+")
    # Insert space after `,!?;:` if missing. For `.` only insert space if the next
    # char is an UPPERCASE letter (sentence boundary). This protects abbreviations
    # ("т.е.") and decimals ("1500.50").
    _RE_NORMALIZE_SPACE_AFTER_PUNCT = re.compile(r"(?<=[,!?;:])(?=[^\s])")
    _RE_NORMALIZE_SPACE_AFTER_DOT = re.compile(r"(?<=\.)(?=[A-ZА-ЯЁ])")
    _RE_NORMALIZE_MULTISPACE = re.compile(r"\s{2,}")
    _RE_TRAILING_PUNCT = re.compile(r"[\s,;:.!?…-]+$")
    _RE_LEADING_PUNCT = re.compile(r"^[\s,;:.!?…-]+")
    _RE_STRIP_CHARS = '"«»„‟\'`()[]{} \t\r\n'

    _RE_TOKEN = re.compile(r"\w+", flags=re.UNICODE)

    # Pattern for ", а " — the dangerous coordinator.
    _RE_A_CONJ = re.compile(r",\s+а\s+")

    def __init__(self, min_content_tokens: int = MIN_CONTENT_TOKENS) -> None:
        self.min_content_tokens = min_content_tokens

    # ---- Public API ------------------------------------------------------

    def split(self, text: str, source_review_id: str | None = None) -> list[Segment]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        hard_chunks = self._hard_split(normalized)

        # Tag chunks whose text begins with a discourse marker (e.g. second sentence
        # starts with "Но" or "В общем,"). These markers carry boundary semantics even
        # when no internal split is needed.
        for ch in hard_chunks:
            self._tag_leading_marker(ch)

        soft_chunks: list[_Chunk] = []
        for hc in hard_chunks:
            soft_chunks.extend(self._soft_split(hc, normalized))

        merged = self._merge_orphans(soft_chunks)

        # Mark START on the very first segment.
        if merged:
            merged[0].boundary_type.add("START")

        return [
            Segment(
                text=ch.text,
                segment_index=i,
                char_start=ch.char_start,
                char_end=ch.char_end,
                source_text=normalized,
                boundary_type=frozenset(ch.boundary_type),
                source_review_id=source_review_id,
            )
            for i, ch in enumerate(merged)
        ]

    @staticmethod
    def _tag_leading_marker(chunk: _Chunk) -> None:
        """If chunk text starts with a discourse marker, add CONTRAST/OPENER to its boundary."""
        head = chunk.text.lower().lstrip()
        for m in CONTRAST_MARKERS:
            if head == m or head.startswith(m + " ") or head.startswith(m + ","):
                chunk.boundary_type.add("CONTRAST")
                return
        for m in OPENER_MARKERS:
            if head == m or head.startswith(m + " ") or head.startswith(m + ","):
                chunk.boundary_type.add("OPENER")
                return

    # ---- Step 1: normalization ------------------------------------------

    def _normalize(self, text: str) -> str:
        s = str(text).replace("\\n", " ").replace("\n", " ").replace("\r", " ")
        s = s.strip(self._RE_STRIP_CHARS)
        # Keep ellipses as a non-terminal pause marker instead of forcing a sentence split.
        s = self._RE_NORMALIZE_ELLIPSIS.sub(" … ", s)
        # Any sequence containing '?' collapses to '?'; pure '!' sequences to '!'.
        s = self._RE_NORMALIZE_INTERROBANG.sub("?", s)
        s = self._RE_NORMALIZE_BANGSEQ.sub("!", s)
        s = self._RE_NORMALIZE_SPACE_AFTER_PUNCT.sub(" ", s)
        s = self._RE_NORMALIZE_SPACE_AFTER_DOT.sub(" ", s)
        s = self._RE_NORMALIZE_MULTISPACE.sub(" ", s)
        return s.strip()

    # ---- Step 2: hard split ---------------------------------------------

    def _hard_split(self, text: str) -> list[_Chunk]:
        """Split on terminal punctuation, protecting abbreviations and decimals."""
        boundaries: list[int] = []  # positions immediately AFTER a hard delimiter
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch in self._TERMINAL_CHARS:
                if ch == "." and self._is_protected_dot(text, i):
                    i += 1
                    continue
                # Boundary right after this char.
                boundaries.append(i + 1)
            i += 1

        # Build chunks between boundaries.
        chunks: list[_Chunk] = []
        prev = 0
        for b in boundaries:
            chunks.append(self._make_chunk(text, prev, b))
            prev = b
        if prev < n:
            chunks.append(self._make_chunk(text, prev, n))

        # Each hard chunk inherits HARD on its left boundary, except the first.
        out: list[_Chunk] = []
        for idx, c in enumerate(chunks):
            if c is None:
                continue
            if idx > 0:
                c.boundary_type.add("HARD")
            out.append(c)
        return out

    def _make_chunk(self, source: str, start: int, end: int) -> _Chunk | None:
        """Slice source[start:end], strip whitespace, return chunk with adjusted offsets, or None if empty."""
        raw = source[start:end]
        lstripped = raw.lstrip()
        new_start = start + (len(raw) - len(lstripped))
        rstripped = lstripped.rstrip()
        new_end = new_start + len(rstripped)
        if not rstripped:
            return None
        return _Chunk(text=rstripped, char_start=new_start, char_end=new_end)

    def _is_protected_dot(self, text: str, idx: int) -> bool:
        """True if the dot at `idx` should NOT trigger a hard boundary."""
        # Decimal: digit . digit
        left = text[idx - 1] if idx > 0 else ""
        right = text[idx + 1] if idx + 1 < len(text) else ""
        if left.isdigit() and right.isdigit():
            return True

        # Initials: single uppercase Cyrillic letter, then dot, then space + uppercase Cyrillic
        if left.isalpha() and left.isupper() and self._is_cyrillic(left):
            # Look at character before `left`
            prev = text[idx - 2] if idx >= 2 else " "
            if not prev.isalpha():
                # right side: could be space + uppercase letter
                rest = text[idx + 1: idx + 4]
                if rest and (rest[0] == " " or rest[0].isupper()):
                    # Look ahead for an uppercase letter
                    j = idx + 1
                    while j < len(text) and text[j] == " ":
                        j += 1
                    if j < len(text) and text[j].isalpha() and text[j].isupper() and self._is_cyrillic(text[j]):
                        return True

        # Abbreviation: check if the dot at `idx` falls on ANY dot position of a
        # known abbreviation. For each abbr, locate its dot offsets, then verify
        # that text[idx - offset : idx + len(abbr) - offset] (case-insensitive)
        # matches the abbr.
        for abbr in ABBREVIATIONS:
            for p, ch in enumerate(abbr):
                if ch != ".":
                    continue
                start = idx - p
                end = start + len(abbr)
                if start < 0 or end > len(text):
                    continue
                window = text[start:end].lower()
                if window != abbr.lower():
                    continue
                # Boundary check: char before `start` (if any) must NOT be a letter
                # (otherwise we matched inside a longer word).
                if start > 0 and text[start - 1].isalpha():
                    continue
                return True
        return False

    @staticmethod
    def _is_cyrillic(ch: str) -> bool:
        return "\u0400" <= ch <= "\u04FF"

    # ---- Step 3: soft split (discourse markers + "а") -------------------

    def _soft_split(self, chunk: _Chunk, source: str) -> list[_Chunk]:
        """Apply discourse-marker and conservative-"а" splits within a hard chunk."""
        cuts: list[tuple[int, str]] = []  # (absolute_position_in_source, reason)

        text = chunk.text
        base = chunk.char_start

        # 1. Contrast and opener markers (single-pass scan; longest-first matching).
        markers_with_reason: list[tuple[str, str]] = (
            [(m, "CONTRAST") for m in CONTRAST_MARKERS]
            + [(m, "OPENER") for m in OPENER_MARKERS]
            + [(m, "LABEL") for m in SECTION_LABEL_MARKERS]
        )
        # Sort longest-first to prevent "в общем" being shadowed by a hypothetical shorter token.
        markers_with_reason.sort(key=lambda x: -len(x[0]))

        seen_positions: set[int] = set()
        for marker, reason in markers_with_reason:
            for pos in self._find_marker_positions(text, marker):
                # pos = position in `text` where the marker word starts.
                # Adjust cut to include any leading ", " before the marker.
                cut_local = self._adjust_cut_for_leading_comma(text, pos)
                if cut_local <= 0:
                    continue
                if cut_local in seen_positions:
                    continue
                # Orphan guards (split-time threshold).
                if not self._passes_split_guard(text[:cut_local], text[cut_local:], reason):
                    continue
                cuts.append((base + cut_local, reason))
                seen_positions.add(cut_local)

        # 2. "а" coordinator (conservative).
        for m in self._RE_A_CONJ.finditer(text):
            comma_pos = m.start()  # position of the comma
            cut_local = comma_pos + 1  # cut AFTER comma → comma stays in LEFT segment
            if cut_local in seen_positions:
                continue
            right_part = text[m.end():]
            # Whitelist check on the very prefix of the right part (including "а").
            right_with_a = text[comma_pos + 1:].lstrip()  # drop comma
            if self._matches_a_whitelist(right_with_a):
                continue
            # Predicate check: at least one predicative token in the right part.
            if not self._has_predicate(right_part):
                continue
            # Orphan guards (split-time threshold).
            if not self._passes_split_guard(text[:cut_local], text[cut_local:], "CONTRAST"):
                continue
            cuts.append((base + cut_local, "CONTRAST"))
            seen_positions.add(cut_local)

        if not cuts:
            return [chunk]

        # Sort cuts by absolute position.
        cuts.sort(key=lambda x: x[0])

        # Build sub-chunks.
        out: list[_Chunk] = []
        prev_abs = chunk.char_start
        prev_reasons: set[str] = set(chunk.boundary_type)  # left boundary of first sub-chunk
        for cut_abs, reason in cuts:
            sub = self._make_chunk(source, prev_abs, cut_abs)
            if sub is not None:
                sub.boundary_type |= prev_reasons
                out.append(sub)
            prev_abs = cut_abs
            prev_reasons = {reason}
        # Tail
        sub = self._make_chunk(source, prev_abs, chunk.char_end)
        if sub is not None:
            sub.boundary_type |= prev_reasons
            out.append(sub)

        return out if out else [chunk]

    def _find_marker_positions(self, text: str, marker: str) -> list[int]:
        """Return positions where `marker` appears as a standalone phrase.

        Marker must be preceded by start-of-string or whitespace/comma,
        and followed by whitespace.
        """
        if marker.endswith(":"):
            pattern = re.compile(
                r"(?:^|(?<=[\s,]))" + re.escape(marker),
                flags=re.IGNORECASE,
            )
            return [m.start() for m in pattern.finditer(text)]
        pattern = re.compile(
            r"(?:^|(?<=[\s,]))" + re.escape(marker) + r"(?=\s)",
            flags=re.IGNORECASE,
        )
        return [m.start() for m in pattern.finditer(text)]

    @staticmethod
    def _adjust_cut_for_leading_comma(text: str, marker_pos: int) -> int:
        """If marker is preceded by ', ', cut right after the comma so it stays in
        the LEFT segment (and the right segment starts cleanly with the marker)."""
        j = marker_pos
        while j > 0 and text[j - 1].isspace():
            j -= 1
        if j > 0 and text[j - 1] == ",":
            return j  # cut AFTER the comma
        return marker_pos

    @staticmethod
    def _matches_a_whitelist(right_part: str) -> bool:
        """True if right_part starts with a phrase from A_NON_SPLIT_PREFIXES."""
        rp = right_part.lower().strip()
        for prefix in A_NON_SPLIT_PREFIXES:
            # Match as a prefix on word boundary.
            if rp == prefix or rp.startswith(prefix + " ") or rp.startswith(prefix + ","):
                return True
        return False

    def _has_predicate(self, text: str) -> bool:
        """True if any token in text has a predicative POS tag."""
        for tok in self._RE_TOKEN.findall(text.lower()):
            if len(tok) < 2:
                continue
            if _is_predicative(tok):
                return True
        return False

    def _has_min_split_content(self, text: str) -> bool:
        return self._content_token_count(text) >= MIN_SPLIT_CONTENT

    def _has_any_content(self, text: str) -> bool:
        return self._content_token_count(text) >= 1

    def _has_min_content(self, text: str) -> bool:
        return self._content_token_count(text) >= self.min_content_tokens

    def _passes_split_guard(self, left: str, right: str, reason: str) -> bool:
        if self._has_min_split_content(left) and self._has_min_split_content(right):
            return True
        if reason == "CONTRAST":
            return self._has_any_content(left) and self._has_any_content(right)
        if reason == "LABEL":
            return self._has_any_content(left) and self._has_any_content(right)
        return False

    def _content_token_count(self, text: str) -> int:
        count = 0
        for tok in self._RE_TOKEN.findall(text.lower()):
            if len(tok) < 3:
                continue
            if tok in CONTENT_STOPLIST:
                continue
            count += 1
        return count

    # ---- Step 4: orphan merge -------------------------------------------

    # Boundary types that mark a chunk as the result of a deliberate discourse split.
    # Such chunks must NOT be folded back by orphan merge.
    _SPLIT_DERIVED_BOUNDARIES: frozenset[str] = frozenset({"CONTRAST", "OPENER", "LABEL"})

    def _is_trivial_tail(self, chunk: _Chunk) -> bool:
        head = self._RE_LEADING_PUNCT.sub("", chunk.text.lower())
        head = self._RE_TRAILING_PUNCT.sub("", head).strip()
        if not head:
            return True
        if self._content_token_count(head) > 0:
            return False
        for marker in TRIVIAL_TAIL_MARKERS:
            if head == marker:
                return True
        return False

    def _is_orphan(self, chunk: _Chunk) -> bool:
        """A chunk is an orphan if it lacks content AND its boundary is not split-derived."""
        if self._is_trivial_tail(chunk):
            return True
        if chunk.boundary_type & self._SPLIT_DERIVED_BOUNDARIES:
            return False
        return self._content_token_count(chunk.text) < self.min_content_tokens

    def _merge_orphans(self, chunks: list[_Chunk]) -> list[_Chunk]:
        if not chunks:
            return []
        if len(chunks) == 1:
            return chunks

        result: list[_Chunk] = []
        buffer = chunks[0]

        for nxt in chunks[1:]:
            nxt_orphan = self._is_orphan(nxt)
            buf_orphan = self._is_orphan(buffer)
            nxt_trivial_tail = self._is_trivial_tail(nxt)
            buf_trivial_tail = self._is_trivial_tail(buffer)

            # Additional protection: if the boundary BETWEEN buffer and nxt is split-derived
            # (i.e. nxt was born from a discourse-marker split), don't merge across that boundary
            # even if buffer happens to be short.
            split_between = bool(nxt.boundary_type & self._SPLIT_DERIVED_BOUNDARIES)
            if nxt_trivial_tail or buf_trivial_tail:
                split_between = False

            if nxt_orphan and not split_between:
                buffer = self._merge_two(buffer, nxt, keep_left_boundary=True)
            elif buf_orphan and not split_between:
                if result:
                    result[-1] = self._merge_two(result[-1], buffer, keep_left_boundary=True)
                    buffer = nxt
                else:
                    buffer = self._merge_two(buffer, nxt, keep_left_boundary=True)
            else:
                result.append(buffer)
                buffer = nxt

        # Flush buffer.
        if self._is_orphan(buffer) and result:
            result[-1] = self._merge_two(result[-1], buffer, keep_left_boundary=True)
        else:
            result.append(buffer)

        return result

    @staticmethod
    def _merge_two(a: _Chunk, b: _Chunk, keep_left_boundary: bool = True) -> _Chunk:
        """Merge two chunks. Text concat with single space. Boundary from `a`."""
        merged_text = (a.text + " " + b.text).strip()
        return _Chunk(
            text=merged_text,
            char_start=a.char_start,
            char_end=b.char_end,
            boundary_type=set(a.boundary_type) if keep_left_boundary else set(b.boundary_type),
        )


# ---------------------------------------------------------------------------
# Inline tests
# ---------------------------------------------------------------------------

def _print_segments(label: str, segments: list[Segment]) -> None:
    print(f"\n--- {label} ---")
    print(f"count: {len(segments)}")
    for s in segments:
        bt = ",".join(sorted(s.boundary_type))
        print(f"  [{s.segment_index}] ({bt}) {s.text!r}")


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    return condition


if __name__ == "__main__":
    seg = RuleBasedClauseSegmenter()

    print("=" * 80)
    print("INLINE TESTS")
    print("=" * 80)

    all_passed = True

    # -------------------------------------------------------------------
    # Test 1: Простой контраст (с достаточным content на обеих сторонах)
    # -------------------------------------------------------------------
    t1 = "номер был чистый, но завтрак скудный"
    s1 = seg.split(t1)
    _print_segments("Test 1: contrast 'но'", s1)
    ok = (
        _check("len == 2", len(s1) == 2)
        and _check("seg[1] starts with 'но'", s1[1].text.lower().startswith("но"))
        and _check("seg[1] has CONTRAST boundary", "CONTRAST" in s1[1].boundary_type)
    )
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 2: Перечисление с "а ещё" — НЕ режем
    # -------------------------------------------------------------------
    t2 = "номер маленький, а ещё там грязно"
    s2 = seg.split(t2)
    _print_segments("Test 2: enumeration 'а ещё' (whitelist)", s2)
    ok = _check("len == 1 (whitelist suppresses split)", len(s2) == 1)
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 3: Контрастное "а" с предикатом — режем
    # -------------------------------------------------------------------
    t3 = "номер чистый, а персонал хамит"
    s3 = seg.split(t3)
    _print_segments("Test 3: contrastive 'а' with predicate", s3)
    ok = (
        _check("len == 2", len(s3) == 2)
        and _check("seg[1] starts with 'а'", s3[1].text.lower().lstrip(", ").startswith("а"))
        and _check("seg[1] has CONTRAST boundary", "CONTRAST" in s3[1].boundary_type)
    )
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 4: Discourse opener
    # -------------------------------------------------------------------
    t4 = "Был в номере целый день. В общем, всё плохо очень."
    s4 = seg.split(t4)
    _print_segments("Test 4: opener 'в общем'", s4)
    ok = (
        _check("len == 2", len(s4) == 2)
        and _check("seg[1] has HARD and OPENER", {"HARD", "OPENER"}.issubset(s4[1].boundary_type))
    )
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 5: Orphan merge
    # -------------------------------------------------------------------
    t5 = "Кошмар. Просто ужас. Не приеду больше никогда сюда."
    s5 = seg.split(t5)
    _print_segments("Test 5: orphan merge", s5)
    # "Кошмар" — 1 content token (orphan).
    # "Просто ужас" — "просто" stop-word? нет, не в стоп-листе; "ужас" — content. 1-2 content tokens.
    # "Не приеду больше никогда сюда" — длинный.
    # Ожидание: первые два сольются в один, длинный отдельно → 2 сегмента.
    ok = _check("len == 2 (short prefixes merged)", len(s5) == 2)
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 6: Аббревиатура т.е. — НЕ режем
    # -------------------------------------------------------------------
    t6 = "номер был мал, т.е. совсем крошечный"
    s6 = seg.split(t6)
    _print_segments("Test 6: abbreviation 'т.е.'", s6)
    ok = _check("len == 1 (abbreviation protected)", len(s6) == 1)
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 7: Десятичная дробь — НЕ режем
    # -------------------------------------------------------------------
    t7 = "цена 1500.50 рублей очень высокая"
    s7 = seg.split(t7)
    _print_segments("Test 7: decimal number", s7)
    ok = _check("len == 1 (decimal protected)", len(s7) == 1)
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # Test 8: Gold case — гостиница
    # -------------------------------------------------------------------
    t8 = (
        "Состояние гостиницы, конечно, оставляет желать лучшего. Но это не главное... "
        "Нет взаимодействия отдела бронирования и лиц, ответственных за заселение. "
        "Мы забронировали номер по телефону на 23 мая, но изменилось время нашего "
        "прилёта на 24 мая. Я ещё до вылета позвонила в отдел бронирования и "
        "сообщила, что изменилась дата приезда, поэтому надо решить вопрос с бронью, "
        "чтоб не остаться без номера. Сотрудник отдел бронирования заверила, что "
        "бронь сохраниться за нами, что волноваться не о чем. Но в день приезда "
        "девушка на рессепшне сказала, что наша бронь сгорела, номер занят, а "
        "свободных нет. И сделала акцент, что надо было платить за бронь. Последнее "
        "нам никто не предлагал сделать! В общем, пришлось срочно искать другую "
        "гостиницу. Что за барак творится в знаменитой гостинице?!"
    )
    s8 = seg.split(t8)
    _print_segments("Test 8: gold case (hotel)", s8)
    # Без жесткого ассерта на число — это качественный тест.
    print(f"  [INFO] gold case produced {len(s8)} segments (target: 8-12)")

    # -------------------------------------------------------------------
    # Coverage check
    # -------------------------------------------------------------------
    print("\n--- Coverage check on gold case ---")
    norm = seg._normalize(t8)
    norm_content = seg._content_token_count(norm)
    seg_content = sum(seg._content_token_count(s.text) for s in s8)
    coverage = seg_content / norm_content if norm_content else 0.0
    print(f"  normalized content tokens: {norm_content}")
    print(f"  segments content tokens:   {seg_content}")
    print(f"  coverage: {coverage:.3f}")
    ok = _check("coverage >= 0.99", coverage >= 0.99, f"got {coverage:.3f}")
    all_passed = all_passed and ok

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Latency benchmark
    # -------------------------------------------------------------------
    print("\n--- Latency benchmark (100 reviews) ---")
    import time
    import statistics
    bench_corpus = [t8] * 100  # use the gold case repeatedly as representative
    # Warm up pymorphy3 cache.
    for _ in range(3):
        seg.split(t8)
    timings_ms: list[float] = []
    for txt in bench_corpus:
        t0 = time.perf_counter()
        seg.split(txt)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    median = statistics.median(timings_ms)
    p95 = timings_ms[int(0.95 * len(timings_ms))]
    p99 = timings_ms[int(0.99 * len(timings_ms))]
    print(f"  median: {median:.2f} ms")
    print(f"  p95:    {p95:.2f} ms")
    print(f"  p99:    {p99:.2f} ms")
    ok = _check("p95 <= 30 ms", p95 <= 30.0, f"got p95={p95:.2f} ms")
    all_passed = all_passed and ok

    print("\n" + "=" * 80)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 80)
