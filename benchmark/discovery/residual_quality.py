from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
import pymorphy3

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.discovery.run_discovery import (
    _load_hybrid_vocabulary,
    _parse_gold_labels,
    _resolve_repo_path,
    _row_to_review_input,
)
from configs.configs import config
from src.discovery import ResidualExtractor, ResidualResult
from src.vocabulary.loader import Vocabulary

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


@dataclass(slots=True)
class ReviewResidualDiagnostic:
    category_id: str
    nm_id: int
    review_id: str
    n_covered_phrases: int
    n_covered_aspects: int
    n_residual_phrases: int
    n_unique_residual_phrases: int
    n_unigram_residual_phrases: int
    n_bigram_residual_phrases: int
    n_uncovered_gold_aspects: int
    residual_hits_uncovered_gold: bool


def _word_count(phrase: str) -> int:
    return len(_TOKEN_RE.findall(str(phrase)))


def _lemmatize_text(text: str, morph: pymorphy3.MorphAnalyzer) -> set[str]:
    lemmas: set[str] = set()
    for token in _TOKEN_RE.findall(str(text).lower()):
        parses = morph.parse(token)
        if parses:
            lemmas.add(str(parses[0].normal_form or token).lower())
        else:
            lemmas.add(token.lower())
    return lemmas


def _build_vocabulary_lemma_sets(
    vocabulary: Vocabulary,
    morph: pymorphy3.MorphAnalyzer,
) -> list[set[str]]:
    lemma_sets: list[set[str]] = []
    for aspect in vocabulary.aspects:
        lemmas = _lemmatize_text(aspect.canonical_name, morph)
        for synonym in aspect.synonyms:
            lemmas.update(_lemmatize_text(synonym, morph))
        lemma_sets.append(lemmas)
    return lemma_sets


def _extract_uncovered_gold_aspects(
    gold_labels: dict[str, float],
    vocabulary_lemma_sets: list[set[str]],
    morph: pymorphy3.MorphAnalyzer,
) -> list[str]:
    uncovered: list[str] = []
    for aspect_name in sorted(str(key).strip() for key in gold_labels if str(key).strip()):
        aspect_lemmas = _lemmatize_text(aspect_name, morph)
        if aspect_lemmas and any(aspect_lemmas & vocab_lemmas for vocab_lemmas in vocabulary_lemma_sets):
            continue
        uncovered.append(aspect_name)
    return uncovered


def _residual_hits_uncovered_gold(
    residual_phrases: list[str],
    uncovered_gold_aspects: list[str],
    morph: pymorphy3.MorphAnalyzer,
) -> bool:
    if not residual_phrases or not uncovered_gold_aspects:
        return False
    residual_lemmas: set[str] = set()
    for phrase in residual_phrases:
        residual_lemmas.update(_lemmatize_text(phrase, morph))
    gold_lemmas: set[str] = set()
    for aspect_name in uncovered_gold_aspects:
        gold_lemmas.update(_lemmatize_text(aspect_name, morph))
    return bool(residual_lemmas & gold_lemmas)


def build_review_diagnostic(
    *,
    category_id: str,
    nm_id: int,
    review_id: str,
    residual: ResidualResult,
    uncovered_gold_aspects: list[str],
    morph: pymorphy3.MorphAnalyzer,
) -> ReviewResidualDiagnostic:
    unigram_count = sum(1 for phrase in residual.residual_phrases if _word_count(phrase) == 1)
    bigram_count = sum(1 for phrase in residual.residual_phrases if _word_count(phrase) == 2)
    return ReviewResidualDiagnostic(
        category_id=category_id,
        nm_id=int(nm_id),
        review_id=str(review_id),
        n_covered_phrases=len(residual.covered_phrases),
        n_covered_aspects=len(residual.covered_aspects),
        n_residual_phrases=len(residual.residual_phrases),
        n_unique_residual_phrases=len(set(residual.residual_phrases)),
        n_unigram_residual_phrases=unigram_count,
        n_bigram_residual_phrases=bigram_count,
        n_uncovered_gold_aspects=len(uncovered_gold_aspects),
        residual_hits_uncovered_gold=_residual_hits_uncovered_gold(
            residual.residual_phrases,
            uncovered_gold_aspects,
            morph,
        ),
    )


def summarize_review_diagnostics(
    diagnostics: list[ReviewResidualDiagnostic],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_category: dict[str, list[ReviewResidualDiagnostic]] = defaultdict(list)
    for item in diagnostics:
        by_category[item.category_id].append(item)

    for category_id, items in sorted(by_category.items()):
        n_reviews = len(items)
        n_residual_reviews = sum(1 for item in items if item.n_residual_phrases > 0)
        n_uncovered_gold_reviews = sum(1 for item in items if item.n_uncovered_gold_aspects > 0)
        n_gold_hit_reviews = sum(
            1
            for item in items
            if item.n_uncovered_gold_aspects > 0 and item.residual_hits_uncovered_gold
        )
        n_residual_phrases = sum(item.n_residual_phrases for item in items)
        n_unigram = sum(item.n_unigram_residual_phrases for item in items)
        n_bigram = sum(item.n_bigram_residual_phrases for item in items)

        rows.append(
            {
                "category_id": category_id,
                "n_reviews": n_reviews,
                "n_reviews_with_residual": n_residual_reviews,
                "residual_review_share": n_residual_reviews / n_reviews if n_reviews else 0.0,
                "n_residual_phrases": n_residual_phrases,
                "mean_residual_phrases_per_review": n_residual_phrases / n_reviews if n_reviews else 0.0,
                "unigram_share": n_unigram / n_residual_phrases if n_residual_phrases else 0.0,
                "bigram_share": n_bigram / n_residual_phrases if n_residual_phrases else 0.0,
                "n_reviews_with_uncovered_gold": n_uncovered_gold_reviews,
                "residual_gold_hit_rate": (
                    n_gold_hit_reviews / n_uncovered_gold_reviews
                    if n_uncovered_gold_reviews
                    else 0.0
                ),
            }
        )
    return rows


def build_top_phrase_rows(
    residuals_by_scope: dict[tuple[str, int | str], list[ResidualResult]],
    *,
    scope_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (category_id, scope_id), residuals in sorted(residuals_by_scope.items()):
        phrase_counter: Counter[str] = Counter()
        review_counter: Counter[str] = Counter()
        for residual in residuals:
            unique_phrases = set()
            for phrase in residual.residual_phrases:
                clean_phrase = phrase.strip()
                if not clean_phrase:
                    continue
                phrase_counter[clean_phrase] += 1
                unique_phrases.add(clean_phrase)
            for phrase in unique_phrases:
                review_counter[phrase] += 1

        total_phrases = sum(phrase_counter.values())
        for phrase, count in phrase_counter.most_common(top_k):
            rows.append(
                {
                    "category_id": category_id,
                    scope_name: scope_id,
                    "phrase": phrase,
                    "count": int(count),
                    "review_count": int(review_counter[phrase]),
                    "word_count": _word_count(phrase),
                    "share_of_residual_phrases": count / total_phrases if total_phrases else 0.0,
                }
            )
    return rows


def render_summary_markdown(
    category_rows: list[dict[str, Any]],
    top_category_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Residual Quality Diagnostic",
        "",
        "## Category Summary",
        "",
        "| category | reviews | residual reviews | residual phrases | unigram share | gold hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in category_rows:
        lines.append(
            "| {category_id} | {n_reviews} | {n_reviews_with_residual} | "
            "{n_residual_phrases} | {unigram_share:.4f} | {residual_gold_hit_rate:.4f} |".format(
                **row
            )
        )

    mean_unigram_share = (
        sum(float(row["unigram_share"]) for row in category_rows) / len(category_rows)
        if category_rows
        else 0.0
    )
    mean_gold_hit_rate = (
        sum(float(row["residual_gold_hit_rate"]) for row in category_rows) / len(category_rows)
        if category_rows
        else 0.0
    )
    verdict = (
        "residual input is weak for HDBSCAN"
        if mean_unigram_share >= 0.6 and mean_gold_hit_rate < 0.4
        else "residual input is not the only bottleneck"
    )

    lines.extend(["", f"Verdict: {verdict}", "", "## Top Residual Phrases", ""])
    for row in top_category_rows[:80]:
        lines.append(
            "- {category_id}: \"{phrase}\" ({count}, reviews={review_count})".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose discovery residual phrase quality.")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=str(config.discovery_runner.dataset_csv),
        help="Path to discovery dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.discovery_runner.results_dir),
        help="Base output directory for diagnostic artifacts.",
    )
    args = parser.parse_args()

    dataset_csv = _resolve_repo_path(args.dataset_csv)
    output_root = _resolve_repo_path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{timestamp}_residual_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    category_column = "category_id" if "category_id" in df.columns else "category"
    if category_column not in df.columns:
        raise ValueError("Dataset must contain `category_id` or `category` column.")

    morph = pymorphy3.MorphAnalyzer()
    extractor = ResidualExtractor(morph=morph)
    diagnostics: list[ReviewResidualDiagnostic] = []
    residuals_by_category: dict[tuple[str, str], list[ResidualResult]] = defaultdict(list)
    residuals_by_product: dict[tuple[str, int], list[ResidualResult]] = defaultdict(list)

    for category_id in list(config.discovery_runner.categories):
        category_df = df[df[category_column].astype(str) == str(category_id)].copy()
        vocabulary = _load_hybrid_vocabulary(str(category_id))
        vocabulary_lemma_sets = _build_vocabulary_lemma_sets(vocabulary, morph)

        for _, row in category_df.iterrows():
            review = _row_to_review_input(row)
            residual = extractor.extract(review, str(category_id), vocabulary)
            gold_labels = _parse_gold_labels(row.get("true_labels"))
            uncovered_gold = _extract_uncovered_gold_aspects(
                gold_labels,
                vocabulary_lemma_sets,
                morph,
            )
            diagnostics.append(
                build_review_diagnostic(
                    category_id=str(category_id),
                    nm_id=int(row["nm_id"]),
                    review_id=str(row["id"]),
                    residual=residual,
                    uncovered_gold_aspects=uncovered_gold,
                    morph=morph,
                )
            )
            residuals_by_category[(str(category_id), "all")].append(residual)
            residuals_by_product[(str(category_id), int(row["nm_id"]))].append(residual)

    review_rows = [asdict(item) for item in diagnostics]
    category_rows = summarize_review_diagnostics(diagnostics)
    top_category_rows = build_top_phrase_rows(
        residuals_by_category,
        scope_name="scope",
        top_k=50,
    )
    top_product_rows = build_top_phrase_rows(
        residuals_by_product,
        scope_name="nm_id",
        top_k=30,
    )

    pd.DataFrame(review_rows).to_csv(output_dir / "residual_review_diagnostics.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output_dir / "residual_summary_by_category.csv", index=False)
    pd.DataFrame(top_category_rows).to_csv(output_dir / "top_residual_phrases_by_category.csv", index=False)
    pd.DataFrame(top_product_rows).to_csv(output_dir / "top_residual_phrases_by_product.csv", index=False)
    (output_dir / "residual_quality_summary.md").write_text(
        render_summary_markdown(category_rows, top_category_rows),
        encoding="utf-8",
    )

    print(f"[residual-quality] Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
