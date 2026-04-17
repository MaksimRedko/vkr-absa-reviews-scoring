from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# При запуске `python benchmark/run_benchmark.py` корень sys.path — папка benchmark/.
# eval_pipeline и run_experiment лежат в корне проекта.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_pipeline import (
    load_markup,
    run_pipeline_for_ids,
    evaluate_with_mapping,
    evaluate_product_ratings,
    _build_auto_mapping,
    MANUAL_MAPPING,
    set_global_seed,
)
from run_experiment import temporary_config_overrides


def _print_summary_table(
    name: str,
    n_venues: int,
    n_reviews: int,
    metrics: Dict[str, Any],
    product_ratings: Dict[str, Any],
) -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"BENCHMARK RESULTS: {name} ({n_venues} venues, {n_reviews} reviews)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'':24s}Per-venue        Global")
    print(
        f"Macro Recall            {metrics.get('macro_recall', 'N/A')}"
    )
    print(
        f"Mention Recall (review) {metrics.get('macro_mention_recall_review', 'N/A')}      "
        f"{metrics.get('global_mention_recall_review', 'N/A')}"
    )
    print(
        f"Sentence MAE (raw)      {metrics.get('macro_mae_raw', 'N/A')}      "
        f"{metrics.get('global_mae_raw', 'N/A')}"
    )
    print(
        f"Product MAE (matched)   "
        f"{product_ratings.get('macro_product_mae', 'N/A')}      "
        f"{product_ratings.get('global_product_mae_filtered', 'N/A')}"
    )
    print()
    print("Per-venue breakdown:")
    per_product = metrics.get("per_product", {}) or {}
    for nm_id, pm in per_product.items():
        recall = pm.get("recall", "N/A")
        mae = pm.get("mae_raw", "N/A")
        print(f"  nm_id={nm_id:<8}  Recall={recall}  MAE={mae}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ABSA pipeline on Yandex Maps benchmark CSV and compute metrics."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Путь к eval CSV (например benchmark/eval_datasets/yandex_maps_benchmark.csv).",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="Тип маппинга аспектов: auto или manual (default: auto).",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.3,
        help="Порог косинусной близости для auto mapping (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для всех генераторов (default: 42).",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    csv_path = args.csv
    df = load_markup(csv_path)
    stats = df.groupby("nm_id").size().reset_index(name="n")
    nm_ids: List[int] = stats["nm_id"].astype(int).tolist()
    n_reviews = int(len(df))

    # Пайплайн без изменения глобальной конфигурации: оборачиваем пустыми overrides
    with temporary_config_overrides({}):
        pipeline_results = run_pipeline_for_ids(
            nm_ids=nm_ids,
            csv_path=csv_path,
            json_path=None,
            fraud_stage=None,
        )

    # Подготовка структуры, аналогичной eval_experiment
    pipeline_results_for_eval: Dict[int, Dict[str, Any]] = {}
    for nm_id, data in pipeline_results.items():
        pipeline_results_for_eval[int(nm_id)] = {
            "aspects": data.get("aspects", []),
            "aspect_keywords": data.get("aspect_keywords", {}),
            "per_review": data.get("per_review", {}),
            "diagnostics": data.get("diagnostics", {}),
        }

    if args.mapping == "auto":
        active_mapping = _build_auto_mapping(
            pipeline_results_for_eval,
            df,
            threshold=args.auto_threshold,
        )
    else:
        active_mapping = MANUAL_MAPPING

    metrics = evaluate_with_mapping(df, pipeline_results_for_eval, active_mapping)
    product_ratings = evaluate_product_ratings(df, pipeline_results_for_eval, active_mapping)

    # Добавляем несколько сводных полей для удобства вывода
    if metrics.get("per_product"):
        metrics["macro_mae_raw"] = float(
            sum(pm["mae_raw"] for pm in metrics["per_product"].values() if pm["mae_raw"] is not None)
        ) / len(metrics["per_product"])
    else:
        metrics["macro_mae_raw"] = None

    results_dir = Path("benchmark") / "eval_datasets" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "benchmark_results.json"
    payload = {
        "csv_path": csv_path,
        "nm_ids": nm_ids,
        "n_reviews": n_reviews,
        "mapping_mode": args.mapping,
        "auto_threshold": args.auto_threshold if args.mapping == "auto" else None,
        "metrics": metrics,
        "product_ratings": product_ratings,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _print_summary_table(
        name="Yandex Maps",
        n_venues=len(nm_ids),
        n_reviews=n_reviews,
        metrics=metrics,
        product_ratings=product_ratings,
    )


if __name__ == "__main__":
    # Гарантируем UTF-8 вывод
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

