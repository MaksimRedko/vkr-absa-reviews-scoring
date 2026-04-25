from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.configs import config
from src.discovery.encoder import DiscoveryEncoder


Pair = tuple[str, str]


SIMILAR_PAIRS: list[Pair] = [
    ("катушка сломалась", "катушка не работает"),
    ("упаковка порвана", "коробка помята"),
    ("быстро пришло", "доставка оперативная"),
    ("кот ест с удовольствием", "хорошо кушает"),
    ("приятный запах", "хорошо пахнет"),
    ("цена завышена", "слишком дорого"),
    ("ткань приятная", "материал мягкий"),
    ("размер маломерит", "маленький размер"),
    ("кнопка заедает", "кнопка плохо нажимается"),
    ("батарея быстро садится", "аккумулятор держит плохо"),
    ("экран яркий", "дисплей очень светлый"),
    ("швы кривые", "пошив неаккуратный"),
    ("сервис ответил быстро", "поддержка среагировала оперативно"),
    ("номер чистый", "в комнате чисто"),
    ("еда вкусная", "блюдо очень вкусное"),
    ("стул устойчивый", "кресло не шатается"),
    ("запах резкий", "сильно пахнет"),
    ("вода горячая", "горячая вода есть"),
    ("ребенок доволен", "ребенку понравилось"),
    ("упаковка надежная", "хорошо упаковано"),
]

DIFFERENT_PAIRS: list[Pair] = [
    ("катушка сломалась", "доставка быстрая"),
    ("упаковка порвана", "вкусно"),
    ("цена высокая", "размер большой"),
    ("материал мягкий", "продавец ответил быстро"),
    ("батарея быстро садится", "номер чистый"),
    ("приятный запах", "доставка задержалась"),
    ("кот ест с удовольствием", "кнопка заедает"),
    ("ткань приятная", "цена завышена"),
    ("еда вкусная", "швы кривые"),
    ("сервис ответил быстро", "экран яркий"),
    ("вода горячая", "упаковка порвана"),
    ("коробка помята", "поддержка вежливая"),
    ("аккумулятор держит плохо", "блюдо вкусное"),
    ("размер маломерит", "запах приятный"),
    ("номер чистый", "доставка оперативная"),
    ("стул устойчивый", "цена завышена"),
    ("кнопка плохо нажимается", "кот хорошо кушает"),
    ("материал мягкий", "вода горячая"),
    ("швы кривые", "приятный запах"),
    ("ребенок доволен", "батарея быстро садится"),
]

RANDOM_NOUN_PAIRS: list[Pair] = [
    ("катушка", "упаковка"),
    ("ребенок", "товар"),
    ("день", "качество"),
    ("доставка", "вкус"),
    ("материал", "цена"),
    ("кнопка", "номер"),
    ("запах", "размер"),
    ("коробка", "поддержка"),
    ("кот", "экран"),
    ("вода", "шов"),
    ("стул", "батарея"),
    ("ткань", "сервис"),
    ("блюдо", "аккумулятор"),
    ("комната", "доставка"),
    ("аппетит", "коробка"),
    ("продавец", "материал"),
    ("цена", "завтрак"),
    ("персонал", "упаковка"),
    ("кресло", "шоколад"),
    ("рыбалка", "подписка"),
]

PAIR_SETS: list[tuple[str, list[Pair]]] = [
    ("similar", SIMILAR_PAIRS),
    ("different", DIFFERENT_PAIRS),
    ("random_nouns", RANDOM_NOUN_PAIRS),
]


@dataclass(frozen=True, slots=True)
class PairRecord:
    model_name: str
    pair_set: str
    pair_index: int
    text_a: str
    text_b: str
    cosine: float


@dataclass(frozen=True, slots=True)
class StatsRow:
    model_name: str
    pair_set: str
    count: int
    min_value: float
    median_value: float
    mean_value: float
    max_value: float
    std_value: float


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _encode_tiny2(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(str(config.models.encoder_path), device="cpu")
    embeddings = np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float32)
    return _l2_normalize(embeddings)


def _encode_sbert_large(texts: list[str]) -> np.ndarray:
    encoder = DiscoveryEncoder(
        model_name_or_path=str(config.discovery_runner.encoder_model),
        batch_size=int(config.discovery_runner.encoder_batch_size),
        device="cpu",
    )
    embeddings = np.asarray(encoder.encode(texts), dtype=np.float32)
    return _l2_normalize(embeddings)


def _collect_pair_records(
    model_name: str,
    encoder_fn: Callable[[list[str]], np.ndarray],
) -> list[PairRecord]:
    all_texts = sorted(
        {
            text
            for _, pairs in PAIR_SETS
            for left, right in pairs
            for text in (left, right)
        }
    )
    embeddings = encoder_fn(all_texts)
    embedding_by_text = {
        text: embeddings[idx]
        for idx, text in enumerate(all_texts)
    }

    records: list[PairRecord] = []
    for pair_set_name, pairs in PAIR_SETS:
        for pair_index, (text_a, text_b) in enumerate(pairs, start=1):
            cosine = float(np.dot(embedding_by_text[text_a], embedding_by_text[text_b]))
            records.append(
                PairRecord(
                    model_name=model_name,
                    pair_set=pair_set_name,
                    pair_index=pair_index,
                    text_a=text_a,
                    text_b=text_b,
                    cosine=cosine,
                )
            )
    return records


def _build_stats(records: list[PairRecord]) -> list[StatsRow]:
    stats: list[StatsRow] = []
    for model_name in ("rubert-tiny2", "sbert_large_nlu_ru"):
        for pair_set in ("similar", "different", "random_nouns"):
            values = np.asarray(
                [
                    record.cosine
                    for record in records
                    if record.model_name == model_name and record.pair_set == pair_set
                ],
                dtype=np.float32,
            )
            stats.append(
                StatsRow(
                    model_name=model_name,
                    pair_set=pair_set,
                    count=int(values.size),
                    min_value=float(values.min()) if values.size else float("nan"),
                    median_value=float(np.median(values)) if values.size else float("nan"),
                    mean_value=float(values.mean()) if values.size else float("nan"),
                    max_value=float(values.max()) if values.size else float("nan"),
                    std_value=float(values.std()) if values.size else float("nan"),
                )
            )
    return stats


def _range_fully_overlapped(similar: StatsRow, different: StatsRow) -> bool:
    return (
        similar.min_value >= different.min_value
        and similar.max_value <= different.max_value
    ) or (
        different.min_value >= similar.min_value
        and different.max_value <= similar.max_value
    )


def _build_verdict(stats: list[StatsRow]) -> tuple[str, float, bool]:
    stats_by_key = {
        (row.model_name, row.pair_set): row
        for row in stats
    }
    similar = stats_by_key[("sbert_large_nlu_ru", "similar")]
    different = stats_by_key[("sbert_large_nlu_ru", "different")]
    separation = similar.median_value - different.median_value
    fully_overlapped = _range_fully_overlapped(similar, different)

    if separation >= 0.20 and not fully_overlapped:
        return ("модель разделяет", separation, fully_overlapped)
    if separation >= 0.10:
        return ("модель разделяет слабо", separation, fully_overlapped)
    return ("модель не разделяет", separation, fully_overlapped)


def _write_csv(output_path: Path, records: list[PairRecord]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "pair_set", "pair_index", "text_a", "text_b", "cosine"])
        for record in records:
            writer.writerow(
                [
                    record.model_name,
                    record.pair_set,
                    record.pair_index,
                    record.text_a,
                    record.text_b,
                    f"{record.cosine:.6f}",
                ]
            )


def _write_summary(
    output_path: Path,
    stats: list[StatsRow],
    verdict: str,
    separation: float,
    fully_overlapped: bool,
) -> None:
    lines = [
        "# Encoder Sanity",
        "",
        f"Вердикт: **{verdict}**",
        f"Разделимость `median(similar) - median(different)` для `sbert_large_nlu_ru`: `{separation:.4f}`",
        f"Полное вложение диапазонов `similar` и `different`: `{fully_overlapped}`",
        "",
        "| Model | Pair set | Count | Min | Median | Mean | Max | Std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats:
        lines.append(
            f"| {row.model_name} | {row.pair_set} | {row.count} | "
            f"{row.min_value:.4f} | {row.median_value:.4f} | {row.mean_value:.4f} | "
            f"{row.max_value:.4f} | {row.std_value:.4f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _print_console(stats: list[StatsRow], verdict: str, separation: float) -> None:
    print("Encoder sanity results")
    print(f"Verdict: {verdict}")
    print(f"Separation: {separation:.4f}")
    print("")
    for model_name in ("rubert-tiny2", "sbert_large_nlu_ru"):
        print(f"Model: {model_name}")
        for pair_set in ("similar", "different", "random_nouns"):
            row = next(
                item
                for item in stats
                if item.model_name == model_name and item.pair_set == pair_set
            )
            print(
                f"  {pair_set:12s} median={row.median_value:.4f} "
                f"range=[{row.min_value:.4f}, {row.max_value:.4f}] std={row.std_value:.4f}"
            )
        print("")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(config.discovery_runner.results_dir).resolve()
        / f"{timestamp}_sanity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[PairRecord] = []
    records.extend(_collect_pair_records("rubert-tiny2", _encode_tiny2))
    records.extend(_collect_pair_records("sbert_large_nlu_ru", _encode_sbert_large))

    stats = _build_stats(records)
    verdict, separation, fully_overlapped = _build_verdict(stats)

    csv_path = output_dir / "pair_similarities.csv"
    summary_path = output_dir / "stats_summary.md"
    _write_csv(csv_path, records)
    _write_summary(summary_path, stats, verdict, separation, fully_overlapped)
    _print_console(stats, verdict, separation)
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
