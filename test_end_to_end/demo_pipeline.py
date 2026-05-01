from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated"
WINDOW_TOKENS = 6
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(slots=True)
class Review:
    review_id: str
    rating: int
    text: str


@dataclass(slots=True)
class Aspect:
    aspect_id: str
    aspect_type: str
    title: str


@dataclass(slots=True)
class Fragment:
    fragment_id: str
    review_id: str
    start_offset: int
    end_offset: int
    offset_status: str
    origin: str


@dataclass(slots=True)
class AspectAssignment:
    assignment_id: str
    review_id: str
    aspect_id: str
    aspect_type: str
    fragment_id: str


@dataclass(slots=True)
class ModeInput:
    assignment_id: str
    review_id: str
    aspect_id: str
    aspect_type: str
    fragment_id: str
    mode_id: str
    premise_text: str


REVIEWS = [
    Review("r01", 2, "Катушка сломалась почти сразу, но доставка была быстрой."),
    Review("r02", 5, "Батарея держит заряд два дня, экран яркий и четкий."),
    Review("r03", 1, "Запах резкий и неприятный, упаковка порвана."),
    Review("r04", 4, "Номер чистый, завтрак вкусный, персонал вежливый."),
    Review("r05", 2, "Ждали мастера сорок минут, зато проблему решили полностью."),
    Review("r06", 5, "Корм пахнет нормально, кот ест с удовольствием."),
    Review("r07", 3, "Доставка обычная, коробка целая, ничего особенного."),
    Review("r08", 1, "Кнопки люфтят, пластик дешёвый, звук тихий."),
    Review("r09", 5, "Сервис ответил быстро, деньги вернули без споров."),
    Review("r10", 4, "Крем впитывается быстро, запах лёгкий и приятный."),
    Review("r11", 2, "Материал колется, швы грубые, но цвет красивый."),
    Review("r12", 5, "Бронь нашли сразу, заселили без очереди, номер просторный."),
    Review("r13", 1, "Соус кислый, банка протекла, вкус испорчен."),
    Review("r14", 4, "Поддержка объяснила всё спокойно, приложение работает стабильно."),
    Review("r15", 2, "Катушка трещит на проводке, ручка болтается."),
    Review("r16", 5, "Мясо свежее, запаха нет, упаковка аккуратная."),
    Review("r17", 1, "Интернет постоянно падает, оператор ничего не сделал."),
    Review("r18", 4, "Постель мягкая, в комнате тихо, уборка аккуратная."),
    Review("r19", 2, "Крышка закрывается туго, инструкция непонятная."),
    Review("r20", 5, "Батарея не греется, зарядка быстрая, корпус приятный."),
]


ASPECTS = [
    Aspect("coil", "vocab", "Катушка"),
    Aspect("delivery", "vocab", "Доставка"),
    Aspect("battery", "vocab", "Батарея"),
    Aspect("screen", "vocab", "Экран"),
    Aspect("smell", "vocab", "Запах"),
    Aspect("packaging", "vocab", "Упаковка"),
    Aspect("cleanliness", "vocab", "Чистота"),
    Aspect("breakfast", "vocab", "Завтрак"),
    Aspect("staff", "vocab", "Персонал"),
    Aspect("waiting_time", "vocab", "Ожидание"),
    Aspect("problem_resolution", "vocab", "Решение проблемы"),
    Aspect("pet_reaction_cluster_1", "discovery", "Кот ест с удовольствием"),
    Aspect("buttons", "vocab", "Кнопки"),
    Aspect("plastic", "vocab", "Пластик"),
    Aspect("sound", "vocab", "Звук"),
    Aspect("refund_cluster_1", "discovery", "Деньги вернули без споров"),
    Aspect("absorption_cluster_1", "discovery", "Впитывается быстро"),
    Aspect("material", "vocab", "Материал"),
    Aspect("seams", "vocab", "Швы"),
    Aspect("checkin_cluster_1", "discovery", "Заселили без очереди"),
    Aspect("space", "vocab", "Простор"),
    Aspect("taste", "vocab", "Вкус"),
    Aspect("support", "vocab", "Поддержка"),
    Aspect("app_stability_cluster_1", "discovery", "Работает стабильно"),
    Aspect("freshness", "vocab", "Свежесть"),
    Aspect("internet", "vocab", "Интернет"),
    Aspect("operator", "vocab", "Оператор"),
    Aspect("bed", "vocab", "Постель"),
    Aspect("silence", "vocab", "Тишина"),
    Aspect("cleaning", "vocab", "Уборка"),
    Aspect("lid", "vocab", "Крышка"),
    Aspect("manual", "vocab", "Инструкция"),
    Aspect("fast_charge_cluster_1", "discovery", "Зарядка быстрая"),
    Aspect("body", "vocab", "Корпус"),
]


FRAGMENT_SPECS = [
    ("f01", "r01", "Катушка сломалась почти сразу", "candidate"),
    ("f02", "r01", "доставка была быстрой", "candidate"),
    ("f03", "r02", "Батарея держит заряд два дня", "candidate"),
    ("f04", "r02", "экран яркий и четкий", "candidate"),
    ("f05", "r03", "Запах резкий и неприятный", "candidate"),
    ("f06", "r03", "упаковка порвана", "candidate"),
    ("f07", "r04", "Номер чистый", "candidate"),
    ("f08", "r04", "завтрак вкусный", "candidate"),
    ("f09", "r04", "персонал вежливый", "candidate"),
    ("f10", "r05", "Ждали мастера сорок минут", "candidate"),
    ("f11", "r05", "проблему решили полностью", "candidate"),
    ("f12", "r06", "Корм пахнет нормально", "candidate"),
    ("f13", "r06", "кот ест с удовольствием", "cluster_anchor"),
    ("f14", "r07", "Доставка обычная", "candidate"),
    ("f15", "r07", "коробка целая", "candidate"),
    ("f16", "r08", "Кнопки люфтят", "candidate"),
    ("f17", "r08", "пластик дешёвый", "candidate"),
    ("f18", "r08", "звук тихий", "candidate"),
    ("f19", "r09", "Сервис ответил быстро", "candidate"),
    ("f20", "r09", "деньги вернули без споров", "cluster_anchor"),
    ("f21", "r10", "впитывается быстро", "cluster_anchor"),
    ("f22", "r10", "запах лёгкий и приятный", "candidate"),
    ("f23", "r11", "Материал колется", "candidate"),
    ("f24", "r11", "швы грубые", "candidate"),
    ("f25", "r12", "Бронь нашли сразу", "candidate"),
    ("f26", "r12", "заселили без очереди", "cluster_anchor"),
    ("f27", "r12", "номер просторный", "candidate"),
    ("f28", "r13", "Соус кислый", "candidate"),
    ("f29", "r13", "банка протекла", "candidate"),
    ("f30", "r13", "вкус испорчен", "candidate"),
    ("f31", "r14", "Поддержка объяснила всё спокойно", "candidate"),
    ("f32", "r14", "приложение работает стабильно", "cluster_anchor"),
    ("f33", "r15", "Катушка трещит", "candidate"),
    ("f34", "r15", "ручка болтается", "candidate"),
    ("f35", "r16", "Мясо свежее", "candidate"),
    ("f36", "r16", "запаха нет", "candidate"),
    ("f37", "r16", "упаковка аккуратная", "candidate"),
    ("f38", "r17", "Интернет постоянно падает", "candidate"),
    ("f39", "r17", "оператор ничего не сделал", "candidate"),
    ("f40", "r18", "Постель мягкая", "candidate"),
    ("f41", "r18", "в комнате тихо", "candidate"),
    ("f42", "r18", "уборка аккуратная", "candidate"),
    ("f43", "r19", "Крышка закрывается туго", "candidate"),
    ("f44", "r19", "инструкция непонятная", "candidate"),
    ("f45", "r20", "Батарея не греется", "candidate"),
    ("f46", "r20", "зарядка быстрая", "cluster_anchor"),
    ("f47", "r20", "корпус приятный", "candidate"),
]


ASSIGNMENT_SPECS = [
    ("a01", "r01", "coil", "vocab", "f01"),
    ("a02", "r01", "delivery", "vocab", "f02"),
    ("a03", "r02", "battery", "vocab", "f03"),
    ("a04", "r02", "screen", "vocab", "f04"),
    ("a05", "r03", "smell", "vocab", "f05"),
    ("a06", "r03", "packaging", "vocab", "f06"),
    ("a07", "r04", "cleanliness", "vocab", "f07"),
    ("a08", "r04", "breakfast", "vocab", "f08"),
    ("a09", "r04", "staff", "vocab", "f09"),
    ("a10", "r05", "waiting_time", "vocab", "f10"),
    ("a11", "r05", "problem_resolution", "vocab", "f11"),
    ("a12", "r06", "smell", "vocab", "f12"),
    ("a13", "r06", "pet_reaction_cluster_1", "discovery", "f13"),
    ("a14", "r07", "delivery", "vocab", "f14"),
    ("a15", "r07", "packaging", "vocab", "f15"),
    ("a16", "r08", "buttons", "vocab", "f16"),
    ("a17", "r08", "plastic", "vocab", "f17"),
    ("a18", "r08", "sound", "vocab", "f18"),
    ("a19", "r09", "support", "vocab", "f19"),
    ("a20", "r09", "refund_cluster_1", "discovery", "f20"),
    ("a21", "r10", "absorption_cluster_1", "discovery", "f21"),
    ("a22", "r10", "smell", "vocab", "f22"),
    ("a23", "r11", "material", "vocab", "f23"),
    ("a24", "r11", "seams", "vocab", "f24"),
    ("a25", "r12", "support", "vocab", "f25"),
    ("a26", "r12", "checkin_cluster_1", "discovery", "f26"),
    ("a27", "r12", "space", "vocab", "f27"),
    ("a28", "r13", "taste", "vocab", "f28"),
    ("a29", "r13", "packaging", "vocab", "f29"),
    ("a30", "r13", "taste", "vocab", "f30"),
    ("a31", "r14", "support", "vocab", "f31"),
    ("a32", "r14", "app_stability_cluster_1", "discovery", "f32"),
    ("a33", "r15", "coil", "vocab", "f33"),
    ("a34", "r15", "buttons", "vocab", "f34"),
    ("a35", "r16", "freshness", "vocab", "f35"),
    ("a36", "r16", "smell", "vocab", "f36"),
    ("a37", "r16", "packaging", "vocab", "f37"),
    ("a38", "r17", "internet", "vocab", "f38"),
    ("a39", "r17", "operator", "vocab", "f39"),
    ("a40", "r18", "bed", "vocab", "f40"),
    ("a41", "r18", "silence", "vocab", "f41"),
    ("a42", "r18", "cleaning", "vocab", "f42"),
    ("a43", "r19", "lid", "vocab", "f43"),
    ("a44", "r19", "manual", "vocab", "f44"),
    ("a45", "r20", "battery", "vocab", "f45"),
    ("a46", "r20", "fast_charge_cluster_1", "discovery", "f46"),
    ("a47", "r20", "body", "vocab", "f47"),
]


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_span(text: str, span_text: str) -> tuple[int, int]:
    start = text.find(span_text)
    if start < 0:
        raise ValueError(f"Фрагмент не найден в отзыве: {span_text!r}")
    return start, start + len(span_text)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in SENTENCE_SPLIT_RE.finditer(text):
        span_end = match.start()
        chunk = text[cursor:span_end].strip()
        if chunk:
            left = text.find(chunk, cursor, span_end)
            spans.append((left, left + len(chunk)))
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        left = text.find(tail, cursor)
        spans.append((left, left + len(tail)))
    if not spans:
        return 0, len(text)
    for left, right in spans:
        if left <= start < right or left < end <= right:
            return left, right
    return spans[0]


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def _window_bounds(text: str, start: int, end: int, window_tokens: int) -> tuple[int, int]:
    spans = _token_spans(text)
    if not spans:
        return 0, len(text)
    anchor_indices = [index for index, (left, right) in enumerate(spans) if left < end and right > start]
    if not anchor_indices:
        return start, end
    left_index = max(0, min(anchor_indices) - window_tokens)
    right_index = min(len(spans) - 1, max(anchor_indices) + window_tokens)
    return spans[left_index][0], spans[right_index][1]


def _build_fragments(reviews_by_id: dict[str, Review]) -> list[Fragment]:
    rows: list[Fragment] = []
    for fragment_id, review_id, span_text, origin in FRAGMENT_SPECS:
        review = reviews_by_id[review_id]
        start, end = _find_span(review.text, span_text)
        rows.append(
            Fragment(
                fragment_id=fragment_id,
                review_id=review_id,
                start_offset=start,
                end_offset=end,
                offset_status="exact",
                origin=origin,
            )
        )
    return rows


def _build_assignments() -> list[AspectAssignment]:
    return [
        AspectAssignment(
            assignment_id=assignment_id,
            review_id=review_id,
            aspect_id=aspect_id,
            aspect_type=aspect_type,
            fragment_id=fragment_id,
        )
        for assignment_id, review_id, aspect_id, aspect_type, fragment_id in ASSIGNMENT_SPECS
    ]


def _restore_mode_inputs(
    *,
    reviews_by_id: dict[str, Review],
    fragments_by_id: dict[str, Fragment],
    assignments: list[AspectAssignment],
    mode_id: str,
) -> list[ModeInput]:
    rows: list[ModeInput] = []
    for assignment in assignments:
        review = reviews_by_id[assignment.review_id]
        fragment = fragments_by_id[assignment.fragment_id]
        if mode_id == "mode_a":
            premise_text = review.text
        elif mode_id == "mode_b":
            left, right = _sentence_bounds(review.text, fragment.start_offset, fragment.end_offset)
            premise_text = review.text[left:right]
        elif mode_id == "mode_c":
            left, right = _window_bounds(review.text, fragment.start_offset, fragment.end_offset, WINDOW_TOKENS)
            premise_text = review.text[left:right]
        else:
            raise ValueError(f"Неизвестный режим: {mode_id}")
        rows.append(
            ModeInput(
                assignment_id=assignment.assignment_id,
                review_id=assignment.review_id,
                aspect_id=assignment.aspect_id,
                aspect_type=assignment.aspect_type,
                fragment_id=assignment.fragment_id,
                mode_id=mode_id,
                premise_text=premise_text,
            )
        )
    return rows


def _restore_entities_from_saved_artifacts() -> tuple[dict[str, Review], dict[str, Fragment], list[AspectAssignment]]:
    reviews = {
        row["review_id"]: Review(
            review_id=str(row["review_id"]),
            rating=int(row["rating"]),
            text=str(row["text"]),
        )
        for row in _load_jsonl(OUT_DIR / "reviews.jsonl")
    }
    fragments = {
        row["fragment_id"]: Fragment(
            fragment_id=str(row["fragment_id"]),
            review_id=str(row["review_id"]),
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            offset_status=str(row["offset_status"]),
            origin=str(row["origin"]),
        )
        for row in _load_jsonl(OUT_DIR / "fragments.jsonl")
    }
    assignments = [
        AspectAssignment(
            assignment_id=str(row["assignment_id"]),
            review_id=str(row["review_id"]),
            aspect_id=str(row["aspect_id"]),
            aspect_type=str(row["aspect_type"]),
            fragment_id=str(row["fragment_id"]),
        )
        for row in _load_jsonl(OUT_DIR / "aspect_assignments.jsonl")
    ]
    return reviews, fragments, assignments


def _write_summary(
    *,
    reviews: list[Review],
    fragments: list[Fragment],
    assignments: list[AspectAssignment],
    mode_a: list[ModeInput],
    mode_b: list[ModeInput],
    mode_c: list[ModeInput],
) -> None:
    lines = [
        "# Demo `test_end_to_end`",
        "",
        "## Что хранится",
        f"- отзывов: {len(reviews)}",
        f"- фрагментов: {len(fragments)}",
        f"- назначений аспектов: {len(assignments)}",
        "",
        "## Проверка честности A/B/C",
        f"- mode_a входов: {len(mode_a)}",
        f"- mode_b входов: {len(mode_b)}",
        f"- mode_c входов: {len(mode_c)}",
        "- ожидается: числа одинаковые",
        "",
        "## Пример восстановления",
        "- `mode_a` берёт весь отзыв по `review_id`",
        "- `mode_b` берёт предложение по `review_id + start_offset/end_offset`",
        "- `mode_c` берёт окно по `review_id + start_offset/end_offset`",
        "",
        "## Важный смысл",
        "- сначала сохраняется одна и та же сущность аспекта в отзыве",
        "- потом разные режимы меняют только контекст",
        "- повторный поиск аспекта по тексту не делается",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reviews_by_id = {review.review_id: review for review in REVIEWS}
    aspects = ASPECTS
    fragments = _build_fragments(reviews_by_id)
    fragments_by_id = {fragment.fragment_id: fragment for fragment in fragments}
    assignments = _build_assignments()

    _save_jsonl(OUT_DIR / "reviews.jsonl", [asdict(item) for item in REVIEWS])
    _save_jsonl(OUT_DIR / "aspects.jsonl", [asdict(item) for item in aspects])
    _save_jsonl(OUT_DIR / "fragments.jsonl", [asdict(item) for item in fragments])
    _save_jsonl(OUT_DIR / "aspect_assignments.jsonl", [asdict(item) for item in assignments])

    restored_reviews_by_id, restored_fragments_by_id, restored_assignments = _restore_entities_from_saved_artifacts()
    mode_a = _restore_mode_inputs(
        reviews_by_id=restored_reviews_by_id,
        fragments_by_id=restored_fragments_by_id,
        assignments=restored_assignments,
        mode_id="mode_a",
    )
    mode_b = _restore_mode_inputs(
        reviews_by_id=restored_reviews_by_id,
        fragments_by_id=restored_fragments_by_id,
        assignments=restored_assignments,
        mode_id="mode_b",
    )
    mode_c = _restore_mode_inputs(
        reviews_by_id=restored_reviews_by_id,
        fragments_by_id=restored_fragments_by_id,
        assignments=restored_assignments,
        mode_id="mode_c",
    )

    assignment_ids = {item.assignment_id for item in restored_assignments}
    if assignment_ids != {item.assignment_id for item in mode_a}:
        raise AssertionError("mode_a потерял назначения")
    if assignment_ids != {item.assignment_id for item in mode_b}:
        raise AssertionError("mode_b потерял назначения")
    if assignment_ids != {item.assignment_id for item in mode_c}:
        raise AssertionError("mode_c потерял назначения")

    _save_jsonl(OUT_DIR / "mode_a_inputs.jsonl", [asdict(item) for item in mode_a])
    _save_jsonl(OUT_DIR / "mode_b_inputs.jsonl", [asdict(item) for item in mode_b])
    _save_jsonl(OUT_DIR / "mode_c_inputs.jsonl", [asdict(item) for item in mode_c])
    _write_summary(
        reviews=REVIEWS,
        fragments=fragments,
        assignments=assignments,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
    )

    print(str(OUT_DIR))


if __name__ == "__main__":
    main()
