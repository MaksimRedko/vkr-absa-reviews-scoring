from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pymorphy3
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vocabulary.loader import AspectDefinition, Vocabulary


_MORPH = pymorphy3.MorphAnalyzer()


@dataclass(frozen=True, slots=True)
class ProductGoldAspects:
    nm_id: int
    category_id: str
    gold_aspects: frozenset[str]


def _normalize_text(text: str) -> str:
    tokens = [token.strip().lower() for token in str(text).replace("-", " ").split() if token.strip()]
    lemmas: list[str] = []
    for token in tokens:
        parses = _MORPH.parse(token)
        lemma = parses[0].normal_form if parses else token
        lemmas.append(str(lemma).lower())
    return " ".join(lemmas)


def _parse_true_labels(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _load_category_map(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    category_path = Path(path)
    payload = yaml.safe_load(category_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Category map must be a mapping: nm_id -> category_id")
    return {int(k): str(v) for k, v in payload.items()}


def _load_gold_from_markup_csv(csv_path: str | Path) -> list[ProductGoldAspects]:
    df = pd.read_csv(csv_path, dtype={"id": str})
    if "nm_id" not in df.columns or "true_labels" not in df.columns:
        raise ValueError("Markup CSV must contain columns 'nm_id' and 'true_labels'")

    category_column = None
    if "category_id" in df.columns:
        category_column = "category_id"
    elif "category" in df.columns:
        category_column = "category"

    product_aspects: dict[int, set[str]] = defaultdict(set)
    category_by_product: dict[int, str] = {}

    for _, row in df.iterrows():
        labels = _parse_true_labels(row.get("true_labels"))
        if not labels:
            continue
        nm_id = int(row["nm_id"])
        product_aspects[nm_id].update(str(k) for k in labels.keys())
        category = row.get(category_column) if category_column else None
        if pd.notna(category) and str(category).strip():
            category_by_product[nm_id] = str(category).strip()

    return [
        ProductGoldAspects(
            nm_id=nm_id,
            category_id=category_by_product.get(nm_id, "unknown"),
            gold_aspects=frozenset(sorted(aspects)),
        )
        for nm_id, aspects in sorted(product_aspects.items())
        if aspects
    ]


def _load_git_file(spec: str) -> str:
    completed = subprocess.run(
        ["git", "show", spec],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    return completed.stdout


def _load_gold_from_benchmark_results_yaml(
    yaml_text: str,
    *,
    category_map: dict[int, str],
) -> list[ProductGoldAspects]:
    payload = yaml.safe_load(yaml_text)
    per_product = (((payload or {}).get("metrics") or {}).get("per_product") or {})
    if not isinstance(per_product, dict):
        raise ValueError("Benchmark results YAML does not contain metrics.per_product")

    rows: list[ProductGoldAspects] = []
    for nm_id_raw, item in sorted(per_product.items(), key=lambda kv: int(kv[0])):
        if not isinstance(item, dict):
            continue
        true_aspects = item.get("true_aspects") or []
        if not isinstance(true_aspects, list):
            continue
        nm_id = int(nm_id_raw)
        rows.append(
            ProductGoldAspects(
                nm_id=nm_id,
                category_id=category_map.get(nm_id, "unknown"),
                gold_aspects=frozenset(str(x) for x in true_aspects if str(x).strip()),
            )
        )
    return rows


def _build_vocabulary_term_index(vocabulary: Vocabulary) -> set[str]:
    terms: set[str] = set()
    for aspect in vocabulary.aspects:
        terms.add(_normalize_text(aspect.canonical_name))
        for synonym in aspect.synonyms:
            terms.add(_normalize_text(synonym))
    return {term for term in terms if term}


def _load_vocabulary(paths: list[str]) -> Vocabulary:
    aspects: list[AspectDefinition] = []
    by_id: dict[str, AspectDefinition] = {}
    by_canonical: dict[str, AspectDefinition] = {}

    for rel_path in paths:
        vocab = Vocabulary.load_from_yaml(ROOT / rel_path)
        for aspect in vocab.aspects:
            if aspect.id in by_id:
                raise ValueError(f"Duplicate aspect id across vocabularies: {aspect.id}")
            aspects.append(aspect)
            by_id[aspect.id] = aspect
            by_canonical[aspect.canonical_name] = aspect

    return Vocabulary(aspects, _by_id=by_id, _by_canonical=by_canonical)


def _evaluate_coverage(
    products: list[ProductGoldAspects],
    vocabulary: Vocabulary,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vocab_terms = _build_vocabulary_term_index(vocabulary)
    product_rows: list[dict[str, Any]] = []

    for product in products:
        covered = sorted(
            aspect for aspect in product.gold_aspects if _normalize_text(aspect) in vocab_terms
        )
        missed = sorted(set(product.gold_aspects) - set(covered))
        total = len(product.gold_aspects)
        product_rows.append(
            {
                "nm_id": product.nm_id,
                "category_id": product.category_id,
                "total_gold_aspects": total,
                "covered_gold_aspects": len(covered),
                "coverage": round(len(covered) / total, 4) if total else 0.0,
                "covered_aspects": json.dumps(covered, ensure_ascii=False),
                "missed_aspects": json.dumps(missed, ensure_ascii=False),
            }
        )

    per_product_df = pd.DataFrame(product_rows)
    if per_product_df.empty:
        return per_product_df, pd.DataFrame(
            columns=["category_id", "n_products", "total_gold_aspects", "covered_gold_aspects", "coverage"]
        )

    category_rows: list[dict[str, Any]] = []
    for category_id, grp in per_product_df.groupby("category_id", dropna=False):
        total_gold = int(grp["total_gold_aspects"].sum())
        covered_gold = int(grp["covered_gold_aspects"].sum())
        category_rows.append(
            {
                "category_id": category_id,
                "n_products": int(len(grp)),
                "total_gold_aspects": total_gold,
                "covered_gold_aspects": covered_gold,
                "coverage": round(covered_gold / total_gold, 4) if total_gold else 0.0,
            }
        )

    category_df = pd.DataFrame(category_rows).sort_values(["category_id"]).reset_index(drop=True)
    return per_product_df.sort_values(["category_id", "nm_id"]).reset_index(drop=True), category_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute vocabulary coverage against gold aspect labels")
    parser.add_argument(
        "--vocab-path",
        action="append",
        required=True,
        help="Path to vocabulary YAML; repeat flag to merge multiple files",
    )
    parser.add_argument(
        "--markup-csv",
        default=None,
        help="Markup CSV with true_labels and optional category_id",
    )
    parser.add_argument(
        "--benchmark-results-yaml",
        default=None,
        help="Local benchmark results YAML with metrics.per_product.*.true_aspects",
    )
    parser.add_argument(
        "--git-benchmark-results",
        default=None,
        help="Read benchmark results YAML via git show, e.g. <rev>:path/to/file.yaml",
    )
    parser.add_argument(
        "--category-map",
        default=None,
        help="Optional YAML/JSON mapping nm_id -> category_id",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for coverage artifacts",
    )
    args = parser.parse_args()

    input_modes = [
        bool(args.markup_csv),
        bool(args.benchmark_results_yaml),
        bool(args.git_benchmark_results),
    ]
    if sum(input_modes) != 1:
        raise SystemExit("Specify exactly one input source: --markup-csv or --benchmark-results-yaml or --git-benchmark-results")

    category_map = _load_category_map(args.category_map)
    vocabulary = _load_vocabulary(args.vocab_path)

    if args.markup_csv:
        products = _load_gold_from_markup_csv(ROOT / args.markup_csv)
        input_description = str(Path(args.markup_csv))
    elif args.benchmark_results_yaml:
        yaml_text = Path(args.benchmark_results_yaml).read_text(encoding="utf-8")
        products = _load_gold_from_benchmark_results_yaml(yaml_text, category_map=category_map)
        input_description = str(Path(args.benchmark_results_yaml))
    else:
        yaml_text = _load_git_file(args.git_benchmark_results)
        products = _load_gold_from_benchmark_results_yaml(yaml_text, category_map=category_map)
        input_description = f"git:{args.git_benchmark_results}"

    if not products:
        raise SystemExit("No gold aspects loaded from the selected input source")

    per_product_df, category_df = _evaluate_coverage(products, vocabulary)

    total_gold = int(per_product_df["total_gold_aspects"].sum()) if not per_product_df.empty else 0
    total_covered = int(per_product_df["covered_gold_aspects"].sum()) if not per_product_df.empty else 0
    overall_coverage = round(total_covered / total_gold, 4) if total_gold else 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_product_df.to_csv(out_dir / "coverage_per_product.csv", index=False, encoding="utf-8")
    category_df.to_csv(out_dir / "coverage_by_category.csv", index=False, encoding="utf-8")

    summary = {
        "input_source": input_description,
        "vocab_paths": args.vocab_path,
        "n_products": int(len(per_product_df)),
        "total_gold_aspects": total_gold,
        "covered_gold_aspects": total_covered,
        "overall_coverage": overall_coverage,
        "categories_present": category_df["category_id"].tolist() if not category_df.empty else [],
    }
    (out_dir / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
