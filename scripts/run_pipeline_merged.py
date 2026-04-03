"""
Полный ABSAPipeline (как на беке: antifraud → candidates → KeyBERT → кластеры → NLI → агрегация)
по отзывам из parser/reviews_batches/merged_checked_reviews.csv (без SQLite).

Запуск из корня репозитория:
  python scripts/run_pipeline_merged.py
  python scripts/run_pipeline_merged.py --csv-path parser/reviews_batches/merged_checked_reviews.csv --out-dir experiments/merged_run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from eval_pipeline import load_pipeline_reviews_from_csv  # noqa: E402
from src.pipeline import ABSAPipeline  # noqa: E402
from src.schemas.models import ReviewInput  # noqa: E402

DEFAULT_CSV = ROOT / "parser" / "reviews_batches" / "merged_checked_reviews.csv"
DEFAULT_NM = [15430704, 619500952, 54581151]


def _result_to_jsonable(r):
    return {
        "product_id": r.product_id,
        "reviews_processed": r.reviews_processed,
        "processing_time": r.processing_time,
        "aspects": r.aspects,
        "aspect_keywords": r.aspect_keywords,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    ap.add_argument(
        "--nm-ids",
        type=int,
        nargs="*",
        default=DEFAULT_NM,
        help="nm_id товаров (по умолчанию три из merged)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "experiments" / "merged_pipeline_run"),
    )
    ap.add_argument(
        "--save-input-snapshot",
        action="store_true",
        help="Сохранять входные отзывы (JSONL) перед обработкой каждого товара",
    )
    ap.add_argument(
        "--snapshot-dir",
        type=str,
        default=str(ROOT / "data" / "snapshots"),
        help="Каталог для snapshot входных отзывов",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"Нет файла: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    if args.save_input_snapshot:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ABSAPipeline(db_path=str(ROOT / "data" / "dataset.db"))

    combined = {}
    manual_lines = [
        "# Список аспектов для ручной Precision (TP / FP / Borderline)",
        "# Вопрос: «Этот аспект реально релевантен для этого товара?»",
        "",
    ]

    n_total = len(args.nm_ids)
    t_run0 = time.perf_counter()
    step_durations: list[float] = []

    for idx, nm_id in enumerate(args.nm_ids, start=1):
        t_nm = time.perf_counter()
        print("\n" + "=" * 70)
        print(
            f"ТОВАР {idx}/{n_total}  nm_id={nm_id}  "
            f"(прошло {time.perf_counter() - t_run0:.0f}s с запуска)"
        )
        if step_durations:
            avg = sum(step_durations) / len(step_durations)
            left = avg * (n_total - idx + 1)
            print(f"       оценка осталось ~{left:.0f}s по среднему времени на товар")
        print(f"[1/7] Загрузка из CSV: nm_id={nm_id}")
        raw = load_pipeline_reviews_from_csv(str(csv_path), [nm_id])
        reviews: list[ReviewInput] = []
        for row in raw:
            try:
                ri = ReviewInput(**row)
                if ri.clean_text:
                    reviews.append(ri)
            except Exception as e:
                print(f"  skip id={row.get('id')}: {e}")

        print(f"       ReviewInput с текстом: {len(reviews)}")
        snapshot_path = None
        if args.save_input_snapshot:
            snapshot_path = str(snapshot_dir / f"input_reviews_nm{nm_id}.jsonl")
        result = pipeline.analyze_reviews_list(
            reviews=reviews,
            product_id=nm_id,
            save_input_snapshot=args.save_input_snapshot,
            input_snapshot_path=snapshot_path,
        )
        step_durations.append(time.perf_counter() - t_nm)
        print(
            f"       ✓ товар {idx}/{n_total} готов за {step_durations[-1]:.1f}s "
            f"(всего {time.perf_counter() - t_run0:.1f}s)"
        )
        combined[str(nm_id)] = _result_to_jsonable(result)

        out_json = out_dir / f"pipeline_full_nm{nm_id}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(_result_to_jsonable(result), f, ensure_ascii=False, indent=2)

        manual_lines.append(f"## nm_id={nm_id}")
        manual_lines.append(f"# Отзывов: {result.reviews_processed}, время: {result.processing_time}s")
        for name, m in sorted(
            result.aspects.items(), key=lambda x: x[1]["score"], reverse=True
        ):
            kw = result.aspect_keywords.get(name, [])[:8]
            manual_lines.append(
                f"- {name}: score={m['score']:.3f} raw_mean={m['raw_mean']:.3f} "
                f"mentions={m['mentions']} | keywords={kw}"
            )
        manual_lines.append("")

    all_path = out_dir / "pipeline_full_all_nm.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    manual_path = out_dir / "aspects_for_manual_precision.txt"
    with open(manual_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manual_lines))

    print("\n" + "=" * 70)
    print(f"Общее время прогона: {time.perf_counter() - t_run0:.1f}s")
    print(f"Сохранено: {all_path}")
    print(f"По товарам: pipeline_full_nm*.json")
    print(f"Для ручной precision: {manual_path}")


if __name__ == "__main__":
    main()
