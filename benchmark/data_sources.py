from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

try:
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover
    ds = None


WB_DB_PATH = Path("data") / "dataset.db"
YM_PARQUET_PATH = Path("benchmark") / "raw" / "yandex_maps_full.parquet"

EDA_DIR = Path("benchmark") / "eda"
EDA_CACHE_DIR = EDA_DIR / "cache"
YM_INDEX_PATH = EDA_CACHE_DIR / "yandex_venues_index.parquet"
WB_INDEX_PATH = EDA_CACHE_DIR / "wb_products_index.parquet"


def ensure_eda_dirs() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    EDA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_venue_key(name_ru: str, address: str, rubrics: str) -> str:
    raw = f"{name_ru}|{address}|{rubrics}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return float(h)


def _skew_from_counts(counts: List[int]) -> float:
    """Sample skewness for rating values 1..5 from grouped counts."""
    total = sum(counts)
    if total < 3:
        return 0.0

    values = [1, 2, 3, 4, 5]
    mean = sum(v * c for v, c in zip(values, counts)) / total

    m2 = sum(c * (v - mean) ** 2 for v, c in zip(values, counts)) / total
    if m2 <= 1e-12:
        return 0.0
    m3 = sum(c * (v - mean) ** 3 for v, c in zip(values, counts)) / total

    g1 = m3 / (m2 ** 1.5)
    # Коррекция Фишера-Пирсона для sample skewness
    correction = math.sqrt(total * (total - 1)) / (total - 2)
    return float(correction * g1)


def _append_rating_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5", "n_reviews"]
    for col in required:
        if col not in df.columns:
            df[col] = 0

    entropies: List[float] = []
    skews: List[float] = []
    non5s: List[float] = []
    for _, row in df.iterrows():
        c1 = int(row["rating_1"])
        c2 = int(row["rating_2"])
        c3 = int(row["rating_3"])
        c4 = int(row["rating_4"])
        c5 = int(row["rating_5"])
        total = int(row["n_reviews"])
        counts = [c1, c2, c3, c4, c5]

        entropies.append(round(_entropy_from_counts(counts), 4))
        skews.append(round(_skew_from_counts(counts), 4))
        non5s.append(round(((total - c5) / total * 100.0) if total else 0.0, 2))

    df["entropy"] = entropies
    df["skew"] = skews
    df["non5_pct"] = non5s
    return df


def get_wb_connection() -> sqlite3.Connection:
    if not WB_DB_PATH.is_file():
        raise FileNotFoundError(f"WB DB not found: {WB_DB_PATH}")
    return sqlite3.connect(WB_DB_PATH)


def build_wb_index(force: bool = False) -> pd.DataFrame:
    ensure_eda_dirs()
    if WB_INDEX_PATH.is_file() and not force:
        return pd.read_parquet(WB_INDEX_PATH)

    query = """
    SELECT
        nm_id,
        COUNT(*) AS n_reviews,
        SUM(
            CASE
                WHEN TRIM(COALESCE(full_text, '')) <> ''
                  OR TRIM(COALESCE(pros, '')) <> ''
                  OR TRIM(COALESCE(cons, '')) <> ''
                THEN 1 ELSE 0
            END
        ) AS n_nonempty,
        SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS rating_1,
        SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) AS rating_2,
        SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) AS rating_3,
        SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) AS rating_4,
        SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) AS rating_5,
        AVG(rating) AS avg_rating,
        MIN(created_date) AS min_date,
        MAX(created_date) AS max_date
    FROM reviews
    GROUP BY nm_id
    ORDER BY n_nonempty DESC, n_reviews DESC
    """
    with get_wb_connection() as conn:
        df = pd.read_sql_query(query, conn)
    df = _append_rating_quality_metrics(df)
    df.to_parquet(WB_INDEX_PATH, index=False)
    return df


def load_wb_index(force_rebuild: bool = False) -> pd.DataFrame:
    df = build_wb_index(force=force_rebuild)
    required = {"entropy", "skew", "non5_pct"}
    if not required.issubset(set(df.columns)):
        df = build_wb_index(force=True)
    return df


def load_wb_reviews(nm_id: int, limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    query = """
    SELECT
        id,
        nm_id,
        rating,
        created_date,
        full_text,
        pros,
        cons
    FROM reviews
    WHERE nm_id = ?
      AND (
        TRIM(COALESCE(full_text, '')) <> ''
        OR TRIM(COALESCE(pros, '')) <> ''
        OR TRIM(COALESCE(cons, '')) <> ''
      )
    ORDER BY created_date
    """
    params: List[Any] = [int(nm_id)]
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

    with get_wb_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def _require_pyarrow_dataset():
    if ds is None:
        raise ImportError("pyarrow.dataset is required for lazy parquet access")
    if not YM_PARQUET_PATH.is_file():
        raise FileNotFoundError(f"YM parquet not found: {YM_PARQUET_PATH}")
    return ds.dataset(YM_PARQUET_PATH, format="parquet")


def _iter_ym_batches(columns: List[str], batch_size: int = 50_000) -> Iterable[pd.DataFrame]:
    dataset = _require_pyarrow_dataset()
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def build_ym_index(force: bool = False, batch_size: int = 50_000) -> pd.DataFrame:
    ensure_eda_dirs()
    if YM_INDEX_PATH.is_file() and not force:
        return pd.read_parquet(YM_INDEX_PATH)

    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "name_ru": "",
            "address": "",
            "rubrics": "",
            "n_reviews": 0,
            "n_nonempty": 0,
            "rating_sum": 0.0,
            "text_len_sum": 0,
            "rating_1": 0,
            "rating_2": 0,
            "rating_3": 0,
            "rating_4": 0,
            "rating_5": 0,
        }
    )

    columns = ["name_ru", "address", "rubrics", "rating", "text"]
    for chunk in _iter_ym_batches(columns=columns, batch_size=batch_size):
        chunk["name_ru"] = chunk["name_ru"].fillna("").astype(str)
        chunk["address"] = chunk["address"].fillna("").astype(str)
        chunk["rubrics"] = chunk["rubrics"].fillna("").astype(str)
        chunk["text"] = chunk["text"].fillna("").astype(str)
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce").fillna(0).astype(int)

        for row in chunk.itertuples(index=False):
            venue_key = make_venue_key(row.name_ru, row.address, row.rubrics)
            item = stats[venue_key]
            item["name_ru"] = row.name_ru
            item["address"] = row.address
            item["rubrics"] = row.rubrics
            item["n_reviews"] += 1

            text = row.text.strip()
            if text:
                item["n_nonempty"] += 1
                item["text_len_sum"] += len(text)

            rating = int(row.rating)
            item["rating_sum"] += rating
            if 1 <= rating <= 5:
                item[f"rating_{rating}"] += 1

    rows = []
    for venue_key, item in stats.items():
        n_reviews = item["n_reviews"]
        n_nonempty = item["n_nonempty"]
        rows.append(
            {
                "venue_key": venue_key,
                "name_ru": item["name_ru"],
                "address": item["address"],
                "rubrics": item["rubrics"],
                "n_reviews": n_reviews,
                "n_nonempty": n_nonempty,
                "avg_rating": round(item["rating_sum"] / n_reviews, 3) if n_reviews else None,
                "avg_text_len": round(item["text_len_sum"] / n_nonempty, 1) if n_nonempty else 0.0,
                "rating_1": item["rating_1"],
                "rating_2": item["rating_2"],
                "rating_3": item["rating_3"],
                "rating_4": item["rating_4"],
                "rating_5": item["rating_5"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["n_nonempty", "n_reviews"], ascending=[False, False])
    df = _append_rating_quality_metrics(df)
    df.to_parquet(YM_INDEX_PATH, index=False)
    return df


def load_ym_index(force_rebuild: bool = False) -> pd.DataFrame:
    df = build_ym_index(force=force_rebuild)
    required = {"entropy", "skew", "non5_pct"}
    if not required.issubset(set(df.columns)):
        df = build_ym_index(force=True)
    return df


def get_ym_venue_meta(venue_key: str) -> Dict[str, Any]:
    df = load_ym_index(force_rebuild=False)
    sub = df[df["venue_key"] == venue_key]
    if sub.empty:
        raise KeyError(f"Unknown venue_key: {venue_key}")
    return sub.iloc[0].to_dict()


def load_ym_reviews(
    venue_key: str,
    limit: int | None = None,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    meta = get_ym_venue_meta(venue_key)
    target_name = str(meta["name_ru"])
    target_address = str(meta["address"])
    target_rubrics = str(meta["rubrics"])

    columns = ["name_ru", "address", "rubrics", "rating", "text"]
    parts: List[pd.DataFrame] = []
    collected = 0

    for chunk in _iter_ym_batches(columns=columns, batch_size=batch_size):
        mask = (
            chunk["name_ru"].fillna("").astype(str).eq(target_name)
            & chunk["address"].fillna("").astype(str).eq(target_address)
            & chunk["rubrics"].fillna("").astype(str).eq(target_rubrics)
        )
        sub = chunk.loc[mask, columns].copy()
        if sub.empty:
            continue

        sub["text"] = sub["text"].fillna("").astype(str)
        sub = sub[sub["text"].str.strip() != ""]
        if sub.empty:
            continue

        parts.append(sub)
        collected += len(sub)
        if limit is not None and collected >= limit:
            break

    if not parts:
        return pd.DataFrame(columns=["id", "rating", "text", "name_ru", "address", "rubrics"])

    df = pd.concat(parts, ignore_index=True)
    if limit is not None:
        df = df.head(limit).copy()

    df = df.reset_index(drop=True)
    df["id"] = [f"ym_{venue_key}_{i+1:04d}" for i in range(len(df))]
    return df[["id", "rating", "text", "name_ru", "address", "rubrics"]]

