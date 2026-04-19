"""
Candidate-level matching pilot.

Шаги алгоритма:
  1. Загрузка данных из трёх источников разметки.
  2. Построение vocabulary centroids (encode synonyms+canonical, L2-norm, mean, L2-norm).
  3. Извлечение кандидатов CandidateExtractor (ngram, без dependency filter).
  4. Cosine-matching + softmax-rank (τ=0.1), порог τ_m=0.15.
  5. Агрегация в predicted aspect-set для каждого отзыва.
  6. Alignment predicted canonical ↔ true aspect name через cosine > 0.75.
  7. Evaluation: per-review P/R/F1, macro-average.
  8. Экспорт 5 артефактов в results/<timestamp>/.

Запуск:
    python -m benchmark.candidate_matching.run_candidate_pilot [--config <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Путь к корню репозитория в sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Отключаем dependency filter до импорта CandidateExtractor,
# чтобы не инициализировать spacy-парсер.
from configs import configs as _cfg_module  # noqa: E402
_cfg_module.config.discovery.dependency_filter_enabled = False  # type: ignore[attr-defined]

from sentence_transformers import SentenceTransformer  # noqa: E402
from src.stages.extraction import CandidateExtractor  # noqa: E402
from src.vocabulary.loader import Vocabulary  # noqa: E402

from benchmark.candidate_matching.data_loaders import (  # noqa: E402
    AnnotatedReview,
    load_all_annotated,
)


# ---------------------------------------------------------------------------
# Загрузка конфига
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Шаг 2: vocabulary centroids
# ---------------------------------------------------------------------------

def build_centroids(
    vocab: Vocabulary,
    model: SentenceTransformer,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
        M       : ndarray [|V| x d], L2-normalized centroids
        asp_ids : list of aspect_id в том же порядке, что строки M
    """
    asp_ids: list[str] = []
    all_terms_per_asp: dict[str, list[str]] = {}
    for a in vocab.aspects:
        terms = list(dict.fromkeys([a.canonical_name] + a.synonyms))
        all_terms_per_asp[a.id] = terms
        asp_ids.append(a.id)

    # Уникальные строки для батчевого encode
    unique_terms = list(dict.fromkeys(t for ts in all_terms_per_asp.values() for t in ts))
    vecs = model.encode(
        unique_terms,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    term_to_vec: dict[str, np.ndarray] = {t: np.asarray(v, dtype=np.float32) for t, v in zip(unique_terms, vecs)}

    centroids = []
    for aid in asp_ids:
        term_vecs = np.stack([term_to_vec[t] for t in all_terms_per_asp[aid]])
        centroid = term_vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / (norm + 1e-9)
        centroids.append(centroid.astype(np.float32))

    M = np.stack(centroids)  # [|V| x d]
    return M, asp_ids


# ---------------------------------------------------------------------------
# Шаг 3-4: извлечение и matching кандидатов
# ---------------------------------------------------------------------------

def _build_extractor(cfg_matching: dict) -> CandidateExtractor:
    nr = tuple(cfg_matching.get("ngram_range", [1, 2]))
    mwl = int(cfg_matching.get("min_word_length", 3))
    ext = CandidateExtractor(ngram_range=nr, min_word_length=mwl)
    ext.dependency_filter_enabled = False  # уже отключено глобально, для надёжности
    return ext


def match_candidates(
    review: AnnotatedReview,
    extractor: CandidateExtractor,
    model: SentenceTransformer,
    M: np.ndarray,
    asp_ids: list[str],
    vocab: Vocabulary,
    tau_softmax: float,
    tau_match: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Returns список словарей для candidate_predictions.csv.
    """
    candidates = extractor.extract(review.text)
    if not candidates:
        return []

    spans = [c.span for c in candidates]
    # Дедупликация по span для эффективного encode
    unique_spans = list(dict.fromkeys(spans))
    span_vecs_arr = model.encode(
        unique_spans,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    span_to_vec: dict[str, np.ndarray] = {
        s: np.asarray(v, dtype=np.float32) for s, v in zip(unique_spans, span_vecs_arr)
    }

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        emb = span_to_vec[cand.span]
        sims = M @ emb  # [|V|]
        probs = softmax(sims / tau_softmax)  # [|V|]
        top_idxs = np.argsort(probs)[::-1]

        top1_id = asp_ids[top_idxs[0]]
        top1_prob = float(probs[top_idxs[0]])
        top2_id = asp_ids[top_idxs[1]] if len(top_idxs) > 1 else None
        top2_prob = float(probs[top_idxs[1]]) if len(top_idxs) > 1 else None

        matched = top1_id if top1_prob > tau_match else None
        matched_can = vocab.get_by_id(top1_id).canonical_name if matched else None

        rows.append({
            "review_id": review.review_id,
            "product_id": review.product_id,
            "candidate_span": cand.span,
            "top1_aspect": top1_id,
            "top1_prob": round(top1_prob, 6),
            "top2_aspect": top2_id,
            "top2_prob": round(top2_prob, 6) if top2_prob is not None else None,
            "matched_aspect_or_null": matched,
            "matched_canonical": matched_can,
        })
    return rows


# ---------------------------------------------------------------------------
# Шаг 6: alignment
# ---------------------------------------------------------------------------

def compute_alignment(
    predicted_canonicals: set[str],
    true_aspect_names: set[str],
    model: SentenceTransformer,
    cosine_threshold: float,
    batch_size: int,
) -> pd.DataFrame:
    """
    Строит полный pairwise alignment: [predicted_canonical × true_aspect_name].
    Возвращает DataFrame с колонками [predicted_canonical, true_aspect_name, cosine, accepted].
    """
    if not predicted_canonicals or not true_aspect_names:
        return pd.DataFrame(columns=["predicted_canonical", "true_aspect_name", "cosine", "accepted"])

    pred_list = sorted(predicted_canonicals)
    true_list = sorted(true_aspect_names)

    pred_vecs = model.encode(pred_list, batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False)
    true_vecs = model.encode(true_list, batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False)

    # cosine similarity matrix [len(pred) x len(true)]
    sim_matrix = np.asarray(pred_vecs) @ np.asarray(true_vecs).T

    rows = []
    for i, p in enumerate(pred_list):
        for j, t in enumerate(true_list):
            cos = float(sim_matrix[i, j])
            rows.append({
                "predicted_canonical": p,
                "true_aspect_name": t,
                "cosine": round(cos, 4),
                "accepted": cos >= cosine_threshold,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Шаг 7: evaluation
# ---------------------------------------------------------------------------

def _set_intersection_via_alignment(
    pred_canonicals: set[str],
    true_aspects: frozenset[str],
    align_df: pd.DataFrame,
) -> tuple[int, int]:
    """
    Returns:
        matched_preds : число predicted-аспектов, для которых нашёлся хоть один true-match
        matched_trues : число true-аспектов, для которых нашёлся хоть один pred-match
    Оба значения ≤ соответствующих |A_pred| и |A_true|, поэтому P,R ≤ 1.
    """
    accepted = align_df[align_df["accepted"]]
    # Быстрый lookup: (predicted_canonical, true_aspect_name)
    accepted_set = set(zip(accepted["predicted_canonical"], accepted["true_aspect_name"]))
    # Добавляем прямые совпадения canonical == true_name (вдруг не попали в alignment)
    for p in pred_canonicals:
        if p in true_aspects:
            accepted_set.add((p, p))

    matched_preds: set[str] = set()
    matched_trues: set[str] = set()
    for pc in pred_canonicals:
        for ta in true_aspects:
            if (pc, ta) in accepted_set:
                matched_preds.add(pc)
                matched_trues.add(ta)

    return len(matched_preds), len(matched_trues)


def evaluate_review(
    pred_canonical_names: set[str],
    true_aspects: frozenset[str],
    align_df: pd.DataFrame,
) -> dict[str, float]:
    """Per-review precision / recall / F1."""
    n_pred = len(pred_canonical_names)
    n_true = len(true_aspects)

    if n_pred == 0 and n_true == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched_count": 0}
    if n_pred == 0 or n_true == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_count": 0}

    matched_preds, matched_trues = _set_intersection_via_alignment(
        pred_canonical_names, true_aspects, align_df
    )
    p = matched_preds / n_pred
    r = matched_trues / n_true
    f1 = 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "matched_count": matched_trues,  # для отчёта: сколько true-аспектов покрыто
    }


# ---------------------------------------------------------------------------
# Шаг 8: per_aspect_stats
# ---------------------------------------------------------------------------

def build_per_aspect_stats(
    review_rows: list[dict[str, Any]],
    vocab: Vocabulary,
    align_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Для каждого aspect_id из vocabulary считает TP/FP/FN и P/R/F1.
    """
    accepted_pairs = set()
    for _, row in align_df[align_df["accepted"]].iterrows():
        accepted_pairs.add((row["predicted_canonical"], row["true_aspect_name"]))

    asp_to_canonical: dict[str, str] = {a.id: a.canonical_name for a in vocab.aspects}
    canonical_to_id: dict[str, str] = {a.canonical_name: a.id for a in vocab.aspects}

    # Сначала построим, чего для каждого отзыва предсказано и истинно
    # Нужны per-review агрегированные данные
    per_review: dict[str, dict] = {}
    for row in review_rows:
        rid = row["review_id"]
        if rid not in per_review:
            per_review[rid] = {
                "pred_ids": set(),
                "true_names": row["true_aspects_set"],
            }
        if row.get("matched_aspect_or_null"):
            per_review[rid]["pred_ids"].add(row["matched_aspect_or_null"])

    # Для aspect-level P/R/F1
    tp: dict[str, int] = {aid: 0 for aid in asp_to_canonical}
    fp: dict[str, int] = {aid: 0 for aid in asp_to_canonical}
    fn: dict[str, int] = {aid: 0 for aid in asp_to_canonical}
    times_predicted: dict[str, int] = {aid: 0 for aid in asp_to_canonical}
    times_true: dict[str, int] = {aid: 0 for aid in asp_to_canonical}

    for _rid, rv in per_review.items():
        pred_ids: set[str] = rv["pred_ids"]
        true_names: frozenset[str] = rv["true_names"]

        for aid in asp_to_canonical:
            canonical = asp_to_canonical[aid]
            # Это аспект предсказан в этом отзыве?
            predicted = aid in pred_ids
            # Это аспект истинен с учётом alignment (canonical → true_name)?
            gt_present = (
                any((canonical, tn) in accepted_pairs for tn in true_names)
                or canonical in true_names
            )

            if predicted:
                times_predicted[aid] += 1
            if gt_present:
                times_true[aid] += 1

            if predicted and gt_present:
                tp[aid] += 1
            elif predicted and not gt_present:
                fp[aid] += 1
            elif not predicted and gt_present:
                fn[aid] += 1

    records = []
    for a in vocab.aspects:
        aid = a.id
        tp_v = tp[aid]
        fp_v = fp[aid]
        fn_v = fn[aid]
        p = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
        r = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
        f1 = 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0
        records.append({
            "aspect_id": aid,
            "canonical_name": a.canonical_name,
            "times_predicted": times_predicted[aid],
            "times_true": times_true[aid],
            "tp": tp_v,
            "fp": fp_v,
            "fn": fn_v,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Экспорт summary.md
# ---------------------------------------------------------------------------

def build_summary(
    review_df: pd.DataFrame,
    per_aspect_df: pd.DataFrame,
    cfg: dict[str, Any],
    n_reviews: int,
    n_candidates: int,
    elapsed_s: float,
) -> str:
    macro_p = review_df["precision"].mean()
    macro_r = review_df["recall"].mean()
    macro_f1 = review_df["f1"].mean()

    wb_df = review_df[review_df["source"] == "wb"]
    yn_df = review_df[review_df["source"] == "yandex"]

    def _fmt(df: pd.DataFrame) -> str:
        if df.empty:
            return "нет данных"
        return (
            f"P={df['precision'].mean():.3f}  "
            f"R={df['recall'].mean():.3f}  "
            f"F1={df['f1'].mean():.3f}  (n={len(df)})"
        )

    top5_best = review_df.nlargest(5, "f1")[["review_id", "product_id", "f1", "precision", "recall"]]
    top5_worst = review_df.nsmallest(5, "f1")[["review_id", "product_id", "f1", "precision", "recall"]]

    top_aspects = per_aspect_df.sort_values("times_predicted", ascending=False).head(5)
    worst_recall = per_aspect_df[per_aspect_df["times_true"] > 0].sort_values("recall").head(5)

    lines = [
        "# Candidate Matching Pilot — Summary",
        "",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Отзывов:** {n_reviews}  |  **Кандидатов извлечено:** {n_candidates}  |  **Время:** {elapsed_s:.0f}s",
        f"**τ_softmax:** {cfg['matching']['tau_softmax']}  |  **τ_match:** {cfg['matching']['tau_match']}  |  **alignment_threshold:** {cfg['alignment']['cosine_threshold']}",
        "",
        "## Macro P/R/F1 (все отзывы)",
        "",
        f"| Метрика | Значение |",
        f"|---------|---------|",
        f"| Precision | {macro_p:.4f} |",
        f"| Recall    | {macro_r:.4f} |",
        f"| F1        | {macro_f1:.4f} |",
        "",
        "## Breakdown WB vs Yandex",
        "",
        f"| Источник | P / R / F1 |",
        f"|----------|------------|",
        f"| WB       | {_fmt(wb_df)} |",
        f"| Yandex   | {_fmt(yn_df)} |",
        "",
        "## Top-5 лучших отзывов (по F1)",
        "",
        top5_best.to_markdown(index=False),
        "",
        "## Top-5 худших отзывов (по F1)",
        "",
        top5_worst.to_markdown(index=False),
        "",
        "## Топ-5 самых предсказываемых аспектов",
        "",
        top_aspects[["aspect_id", "canonical_name", "times_predicted", "times_true", "precision", "recall", "f1"]].to_markdown(index=False),
        "",
        "## Топ-5 аспектов с худшим recall (среди присутствующих в gold)",
        "",
        worst_recall[["aspect_id", "canonical_name", "times_true", "times_predicted", "precision", "recall", "f1"]].to_markdown(index=False),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str | Path) -> None:
    import time
    t_start = time.time()

    cfg = _load_config(config_path)
    repo_root = _REPO_ROOT

    print("[1/7] Загрузка данных...")
    reviews = load_all_annotated(
        wb_checked_path=repo_root / cfg["data_sources"]["wb_checked"],
        wb_merged_path=repo_root / cfg["data_sources"]["wb_merged"],
        yandex_benchmark_path=repo_root / cfg["data_sources"]["yandex_benchmark"],
        wb_product_ids=list(cfg["products"]["wb"]),
        yandex_product_ids=list(cfg["products"]["yandex"]),
    )
    print(f"    Загружено {len(reviews)} отзывов из {len({r.product_id for r in reviews})} продуктов")

    print("[2/7] Загрузка модели и vocabulary...")
    model = SentenceTransformer(
        str(repo_root / cfg["encoder_model_path"]),
        device="cpu",
    )
    vocab = Vocabulary.load_from_yaml(repo_root / cfg["vocab_path"])
    print(f"    Vocabulary: {len(vocab.aspects)} аспектов")

    print("[3/7] Построение vocabulary centroids...")
    M, asp_ids = build_centroids(vocab, model, batch_size=cfg["matching"]["batch_size"])
    asp_canonical = [vocab.get_by_id(aid).canonical_name for aid in asp_ids]
    print(f"    Матрица M: {M.shape}")

    print("[4/7] Извлечение и matching кандидатов...")
    extractor = _build_extractor(cfg["matching"])
    tau_s = float(cfg["matching"]["tau_softmax"])
    tau_m = float(cfg["matching"]["tau_match"])
    batch_sz = int(cfg["matching"]["batch_size"])

    all_cand_rows: list[dict[str, Any]] = []
    for review in tqdm(reviews, desc="matching", ncols=90):
        rows = match_candidates(
            review=review,
            extractor=extractor,
            model=model,
            M=M,
            asp_ids=asp_ids,
            vocab=vocab,
            tau_softmax=tau_s,
            tau_match=tau_m,
            batch_size=batch_sz,
        )
        # Добавляем true_aspects_set для дальнейшего использования
        for r in rows:
            r["true_aspects_set"] = review.true_aspects
            r["source"] = review.source
        all_cand_rows.extend(rows)

    print(f"    Кандидатов всего: {len(all_cand_rows)}")
    n_matched = sum(1 for r in all_cand_rows if r["matched_aspect_or_null"])
    print(f"    Из них matched (τ_m={tau_m}): {n_matched}")

    print("[5/7] Alignment predicted ↔ true...")
    # Сбор уникальных predicted canonical names и true aspect names
    pred_canonicals_all: set[str] = set()
    true_names_all: set[str] = set()
    for r in all_cand_rows:
        if r["matched_canonical"]:
            pred_canonicals_all.add(r["matched_canonical"])
        true_names_all.update(r["true_aspects_set"])

    align_df = compute_alignment(
        predicted_canonicals=pred_canonicals_all,
        true_aspect_names=true_names_all,
        model=model,
        cosine_threshold=float(cfg["alignment"]["cosine_threshold"]),
        batch_size=batch_sz,
    )
    n_accepted = align_df["accepted"].sum()
    print(f"    Пар alignment: {len(align_df)}, принято: {n_accepted}")

    print("[6/7] Evaluation per review...")
    # Агрегируем предсказания по review
    from collections import defaultdict
    rev_pred: dict[str, set[str]] = defaultdict(set)
    rev_meta: dict[str, dict] = {}
    for r in all_cand_rows:
        rid = r["review_id"]
        if r["matched_canonical"]:
            rev_pred[rid].add(r["matched_canonical"])
        if rid not in rev_meta:
            rev_meta[rid] = {
                "product_id": r["product_id"],
                "true_aspects": r["true_aspects_set"],
                "source": r["source"],
            }

    # Отзывы, у которых не было кандидатов — тоже включаем
    for review in reviews:
        if review.review_id not in rev_meta:
            rev_meta[review.review_id] = {
                "product_id": review.product_id,
                "true_aspects": review.true_aspects,
                "source": review.source,
            }

    review_rows: list[dict[str, Any]] = []
    for rid, meta in rev_meta.items():
        pred_set = rev_pred.get(rid, set())
        true_set = meta["true_aspects"]
        ev = evaluate_review(pred_set, true_set, align_df)
        review_rows.append({
            "review_id": rid,
            "product_id": meta["product_id"],
            "source": meta["source"],
            "predicted_aspects_json": json.dumps(sorted(pred_set), ensure_ascii=False),
            "true_aspects_json": json.dumps(sorted(true_set), ensure_ascii=False),
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1": ev["f1"],
            "matched_count": ev["matched_count"],
        })

    review_df = pd.DataFrame(review_rows)

    # Добавляем true_aspects_set для per_aspect_stats (отдельная структура)
    cand_rows_with_meta = []
    for r in all_cand_rows:
        cand_rows_with_meta.append(r)

    per_aspect_df = build_per_aspect_stats(
        review_rows=[
            {**{k: v for k, v in r.items() if k != "true_aspects_set"},
             "true_aspects_set": r["true_aspects_set"],
             "matched_aspect_or_null": r.get("matched_aspect_or_null")}
            for r in all_cand_rows
        ],
        vocab=vocab,
        align_df=align_df,
    )

    elapsed = time.time() - t_start
    macro_f1 = review_df["f1"].mean()
    macro_p = review_df["precision"].mean()
    macro_r = review_df["recall"].mean()
    print(f"    Macro P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")

    print("[7/7] Экспорт артефактов...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / cfg["output_dir"] / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # candidate_predictions.csv (без служебных полей)
    cand_export_cols = [
        "review_id", "product_id", "candidate_span",
        "top1_aspect", "top1_prob", "top2_aspect", "top2_prob",
        "matched_aspect_or_null",
    ]
    cand_df = pd.DataFrame([
        {k: r.get(k) for k in cand_export_cols} for r in all_cand_rows
    ])
    cand_df.to_csv(out_dir / "candidate_predictions.csv", index=False)

    # review_predictions.csv
    review_export_cols = [
        "review_id", "product_id", "predicted_aspects_json",
        "true_aspects_json", "precision", "recall", "f1", "matched_count",
    ]
    review_df[review_export_cols].to_csv(out_dir / "review_predictions.csv", index=False)

    # alignment_map.csv
    align_df.to_csv(out_dir / "alignment_map.csv", index=False)

    # per_aspect_stats.csv
    per_aspect_df.to_csv(out_dir / "per_aspect_stats.csv", index=False)

    # summary.md
    summary_text = build_summary(
        review_df=review_df,
        per_aspect_df=per_aspect_df,
        cfg=cfg,
        n_reviews=len(reviews),
        n_candidates=len(all_cand_rows),
        elapsed_s=elapsed,
    )
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"Результаты: {out_dir}")
    print(f"Macro P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"Время: {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candidate-level matching pilot")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "pilot_config.yaml"),
        help="Путь к YAML-конфигу пилота",
    )
    args = parser.parse_args()
    run(args.config)
