from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset


RAW_DIR = Path("benchmark") / "raw"
REVIEWS_BY_VENUE_DIR = RAW_DIR / "reviews_by_venue"


RUBRIC_GROUPS: Dict[str, str] = {
    # Рестораны / кафе / фастфуд
    "Кафе": "food",
    "Ресторан": "food",
    "Столовая": "food",
    "Быстрое питание": "food",
    "Пиццерия": "food",
    # Отели / хостелы
    "Гостиница": "hotel",
    "Хостел": "hotel",
    "Отель": "hotel",
    # Салоны красоты / барбершопы / косметология / ногти
    "Салон красоты": "beauty",
    "Косметология": "beauty",
    "Ногтевая студия": "beauty",
    "Барбершоп": "beauty",
    "Парикмахерская": "beauty",
    # Автосервисы / шиномонтаж / автомойка
    "Автосервис": "auto",
    "Шиномонтаж": "auto",
    "Автомойка": "auto",
    "Автомобильные грузоперевозки": "auto",
    # Медицинские клиники / стоматологии / лаборатории
    "Стоматологическая клиника": "medical",
    "Медицинская лаборатория": "medical",
    "Диагностический центр": "medical",
    "Клиника": "medical",
    # Фитнес-клубы / спортзалы / спа
    "Фитнес-клуб": "fitness",
    "Спортзал": "fitness",
    "Спортивный комплекс": "fitness",
    "Спа-салон": "fitness",
    # Магазины (продуктовые / одежда / др.)
    "Супермаркет": "shop",
    "Магазин продуктов": "shop",
    "Магазин": "shop",
    "Магазин обуви": "shop",
    "Магазин одежды": "shop",
}


TARGET_COUNTS = {
    "food": 2,
    "hotel": 1,
    "beauty": 1,
    "auto": 1,
    "medical": 1,
    "fitness": 1,
    "shop": 1,
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_BY_VENUE_DIR.mkdir(parents=True, exist_ok=True)


def load_full_dataset() -> pd.DataFrame:
    print("[download_yandex_maps] Loading dataset d0rj/geo-reviews-dataset-2023 ...")
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    df = ds.to_pandas()
    return df


def save_full_parquet(df: pd.DataFrame) -> Path:
    out_path = RAW_DIR / "yandex_maps_full.parquet"
    print(f"[download_yandex_maps] Saving full dataset to {out_path} ...")
    df.to_parquet(out_path, index=False)
    return out_path


def _coarse_group_for_rubrics(rubrics: str) -> str | None:
    if not isinstance(rubrics, str) or not rubrics.strip():
        return None
    parts = [r.strip() for r in rubrics.split(";") if r.strip()]
    for part in parts:
        if part in RUBRIC_GROUPS:
            return RUBRIC_GROUPS[part]
    return None


def select_candidate_venues(
    df: pd.DataFrame,
    rng: random.Random,
    max_venues: int | None,
) -> Tuple[List[Dict], Dict[str, List[int]]]:
    print("[download_yandex_maps] Selecting candidate venues ...")

    df = df.copy()
    df["rubric_group"] = df["rubrics"].apply(_coarse_group_for_rubrics)

    # Предфильтрация по тексту и рейтингу
    def _valid_text(t: str) -> bool:
        if not isinstance(t, str):
            return False
        return len(t.strip()) > 20

    mask_valid = df["text"].apply(_valid_text) & df["rating"].between(1, 5)
    df_valid = df[mask_valid].copy()

    # Группируем по (name_ru, address)
    groups = []
    for (name, address), grp in df_valid.groupby(["name_ru", "address"]):
        ratings = grp["rating"].astype(float)
        n = len(grp)
        if n < 100:
            continue
        if ratings.nunique() <= 1:
            continue

        rubric_group = grp["rubric_group"].mode(dropna=True)
        rg = rubric_group.iloc[0] if not rubric_group.empty else None
        if rg is None:
            continue

        groups.append(
            {
                "name_ru": name,
                "address": address,
                "rubric_group": rg,
                "rubrics_raw": list(sorted(set(grp["rubrics"].dropna().tolist()))),
                "indices": grp.index.to_list(),
            }
        )

    print(f"[download_yandex_maps] Candidate venue groups after filtering: {len(groups)}")

    # Перемешиваем для случайного выбора
    rng.shuffle(groups)

    selected: List[Dict] = []
    per_group_counter: Dict[str, int] = defaultdict(int)
    indices_by_key: Dict[str, List[int]] = {}

    for g in groups:
        rg = g["rubric_group"]
        limit = TARGET_COUNTS.get(rg)
        if limit is None:
            continue
        if per_group_counter[rg] >= limit:
            continue
        selected.append(g)
        per_group_counter[rg] += 1
        key = f"{g['name_ru']}|{g['address']}"
        indices_by_key[key] = g["indices"]

        if max_venues is not None and len(selected) >= max_venues:
            break

    print(
        "[download_yandex_maps] Selected venues per group:",
        {k: int(v) for k, v in per_group_counter.items()},
    )
    print(f"[download_yandex_maps] Total selected venues: {len(selected)}")
    return selected, indices_by_key


def materialize_selected_venues(
    df: pd.DataFrame,
    selected_groups: List[Dict],
    indices_by_key: Dict[str, List[int]],
    reviews_per_venue: int,
    seed: int,
) -> None:
    rng = np.random.RandomState(seed)

    venues_summary: List[Dict] = []

    for vidx, g in enumerate(selected_groups, start=1):
        venue_id = f"venue_{vidx:03d}"
        key = f"{g['name_ru']}|{g['address']}"
        idx_list = indices_by_key[key]
        grp = df.loc[idx_list].copy()

        # Берём только валидные отзывы (повторная защита)
        mask_valid = grp["text"].apply(lambda t: isinstance(t, str) and len(t.strip()) > 20)
        grp = grp[mask_valid]
        if grp.empty:
            continue

        if len(grp) <= reviews_per_venue:
            sampled = grp
        else:
            sampled = grp.sample(n=reviews_per_venue, random_state=rng)

        sampled = sampled.reset_index(drop=True)

        rating_counts = Counter(int(r) for r in sampled["rating"])
        rating_distribution = {str(k): int(v) for k, v in sorted(rating_counts.items())}

        venues_summary.append(
            {
                "venue_id": venue_id,
                "name": str(g["name_ru"]),
                "rubric": g["rubric_group"],
                "address": str(g["address"]),
                "n_reviews": int(len(sampled)),
                "rating_distribution": rating_distribution,
                "rubrics_raw": g.get("rubrics_raw", []),
            }
        )

        reviews_payload = []
        for i, row in sampled.iterrows():
            review_id = f"ym_{venue_id}_{i+1:04d}"
            reviews_payload.append(
                {
                    "id": review_id,
                    "venue_id": venue_id,
                    "rating": int(row["rating"]),
                    "text": str(row["text"]),
                }
            )

        out_reviews_path = REVIEWS_BY_VENUE_DIR / f"{venue_id}.json"
        out_reviews_path.parent.mkdir(parents=True, exist_ok=True)
        with out_reviews_path.open("w", encoding="utf-8") as f:
            json.dump(reviews_payload, f, ensure_ascii=False, indent=2)
        print(
            f"[download_yandex_maps] Saved {len(reviews_payload)} reviews to {out_reviews_path}"
        )

    selected_path = RAW_DIR / "selected_venues.json"
    with selected_path.open("w", encoding="utf-8") as f:
        json.dump(venues_summary, f, ensure_ascii=False, indent=2)
    print(f"[download_yandex_maps] Selected venues metadata written to {selected_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and filter Yandex Maps reviews benchmark subset."
    )
    parser.add_argument(
        "--max-venues",
        type=int,
        default=None,
        help="Максимальное число заведений (по умолчанию: все, что удовлетворяют критериям).",
    )
    parser.add_argument(
        "--reviews-per-venue",
        type=int,
        default=100,
        help="Число отзывов на каждое заведение (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайного выбора заведений и отзывов (default: 42).",
    )
    args = parser.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)

    df = load_full_dataset()
    save_full_parquet(df)

    selected_groups, indices_by_key = select_candidate_venues(
        df,
        rng=rng,
        max_venues=args.max_venues,
    )
    if not selected_groups:
        print("[download_yandex_maps] No venues selected — check filters or dataset.")
        return

    materialize_selected_venues(
        df,
        selected_groups=selected_groups,
        indices_by_key=indices_by_key,
        reviews_per_venue=args.reviews_per_venue,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

