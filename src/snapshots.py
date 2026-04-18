"""
snapshots.py — Сохранение и загрузка снепшотов каждой стадии пайплайна.

SnapshotWriter  — пишет JSON-снепшот после каждой стадии.
load_*          — десериализуют снепшот обратно в типизированные объекты,
                  чтобы можно было запустить пайплайн с любой промежуточной точки.

Структура папки снепшотов для одного запуска:
  {run_dir}/snapshots/nm{product_id}/
    00_reviews.jsonl           — входные отзывы (ReviewInput)
    01_fraud.json              — trust_weights + статистика
    02_candidates.json         — List[Candidate] с привязкой к review_id
    03_scored.json             — List[ScoredCandidate] (с эмбеддингами опционально)
    04_clusters.json           — Dict[str, AspectInfo]
    05_sentiment_pairs.json    — NLI-пары до инференса
    06_sentiment_results.json  — List[SentimentResult]
    07_aggregation_input.json  — вход для math_engine
    08_pipeline_result.json    — финальный PipelineResult

Replay с любой стадии:
    scored = load_scored_snapshot("path/to/03_scored.json")
    # → List[ScoredCandidate] с восстановленными np.ndarray эмбеддингами
    aspects = clusterer.cluster(scored)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── JSON-сериализация numpy ───────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _dump(data: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_json_default)


# ── SnapshotWriter ────────────────────────────────────────────────────────────

class SnapshotWriter:
    """
    Записывает снепшоты стадий в папку {base_dir}/nm{product_id}/.

    Параметры:
        base_dir        — корневая папка снепшотов (обычно run_dir/snapshots)
        product_id      — nm_id товара (для изоляции по продуктам)
        save_embeddings — сохранять ли эмбеддинги (увеличивают размер файлов,
                          но нужны для replay с scored/clusters стадии)

    Пример:
        writer = SnapshotWriter(exp.run_dir / "snapshots", nm_id=15430704)
        writer.save_fraud(review_ids, trust_weights)
        writer.save_candidates(review_candidates_map)
        ...
    """

    def __init__(
        self,
        base_dir: Path,
        product_id: int,
        save_embeddings: bool = True,
    ) -> None:
        self.dir = Path(base_dir) / f"nm{product_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_embeddings = save_embeddings
        self.product_id = product_id

    def _path(self, filename: str) -> Path:
        return self.dir / filename

    # ── Стадия 0: Входные отзывы ──────────────────────────────────────────────

    def save_reviews(self, reviews: list) -> None:
        """Сохраняет ReviewInput-объекты в JSONL (1 отзыв = 1 строка)."""
        path = self._path("00_reviews.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in reviews:
                row = {
                    "id": r.id,
                    "nm_id": r.nm_id,
                    "rating": r.rating,
                    "created_date": r.created_date.isoformat() if hasattr(r.created_date, "isoformat") else str(r.created_date),
                    "full_text": r.full_text or "",
                    "pros": r.pros or "",
                    "cons": r.cons or "",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Стадия 1: Fraud ───────────────────────────────────────────────────────

    def save_fraud(self, review_ids: List[str], trust_weights: List[float]) -> None:
        weights = [float(w) for w in trust_weights]
        data = {
            "stage": "01_fraud",
            "nm_id": self.product_id,
            "n_reviews": len(weights),
            "review_ids": review_ids,
            "trust_weights": weights,
            "stats": {
                "min": round(min(weights), 4) if weights else None,
                "max": round(max(weights), 4) if weights else None,
                "mean": round(float(np.mean(weights)), 4) if weights else None,
                "low_trust_count": sum(1 for w in weights if w < 0.3),
            },
        }
        _dump(data, self._path("01_fraud.json"))

    # ── Стадия 2: Кандидаты ───────────────────────────────────────────────────

    def save_candidates(self, review_candidates_map: Dict[str, list]) -> None:
        """
        review_candidates_map: {review_id: List[Candidate]}
        """
        per_review = {}
        total = 0
        for rev_id, cands in review_candidates_map.items():
            per_review[rev_id] = [
                {"span": c.span, "sentence": c.sentence, "token_indices": list(c.token_indices)}
                for c in cands
            ]
            total += len(cands)

        data = {
            "stage": "02_candidates",
            "nm_id": self.product_id,
            "total_candidates": total,
            "per_review": per_review,
        }
        _dump(data, self._path("02_candidates.json"))

    # ── Стадия 3: Scored candidates ───────────────────────────────────────────

    def save_scored(self, scored_candidates: list) -> None:
        items = []
        for s in scored_candidates:
            item: Dict[str, Any] = {
                "span": s.span,
                "score": round(float(s.score), 6),
                "sentence": s.sentence,
            }
            if self.save_embeddings and s.embedding is not None:
                item["embedding"] = s.embedding.tolist() if hasattr(s.embedding, "tolist") else list(s.embedding)
            items.append(item)

        data = {
            "stage": "03_scored",
            "nm_id": self.product_id,
            "total_scored": len(items),
            "save_embeddings": self.save_embeddings,
            "candidates": items,
        }
        _dump(data, self._path("03_scored.json"))

    # ── Стадия 4: Кластеры ────────────────────────────────────────────────────

    def save_clusters(self, aspects: Dict[str, Any]) -> None:
        """aspects: Dict[str, AspectInfo]"""
        serialized = {}
        for name, info in aspects.items():
            entry: Dict[str, Any] = {
                "keywords": list(info.keywords),
                "keyword_weights": list(info.keyword_weights) if info.keyword_weights else [],
                "nli_label": info.nli_label or "",
            }
            if self.save_embeddings and info.centroid_embedding is not None:
                emb = info.centroid_embedding
                entry["centroid_embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            serialized[name] = entry

        data = {
            "stage": "04_clusters",
            "nm_id": self.product_id,
            "n_aspects": len(serialized),
            "aspect_names": sorted(serialized.keys()),
            "aspects": serialized,
        }
        _dump(data, self._path("04_clusters.json"))

    # ── Стадия 5: NLI-пары ────────────────────────────────────────────────────

    def save_sentiment_pairs(self, pairs: list) -> None:
        """pairs: List[Tuple[review_id, sentence, aspect, nli_label, weight]]"""
        items = [
            {
                "review_id": p[0],
                "sentence": p[1],
                "aspect": p[2],
                "nli_label": p[3],
                "weight": round(float(p[4]), 6),
            }
            for p in pairs
        ]
        data = {
            "stage": "05_sentiment_pairs",
            "nm_id": self.product_id,
            "n_pairs": len(items),
            "pairs": items,
        }
        _dump(data, self._path("05_sentiment_pairs.json"))

    # ── Стадия 6: Sentiment results ───────────────────────────────────────────

    def save_sentiment_results(self, results: list) -> None:
        items = [
            {
                "review_id": r.review_id,
                "aspect": r.aspect,
                "sentence": r.sentence,
                "score": round(float(r.score), 4),
                "p_ent_pos": round(float(r.p_ent_pos), 4),
                "p_ent_neg": round(float(r.p_ent_neg), 4),
                "confidence": round(float(r.confidence), 4),
            }
            for r in results
        ]
        data = {
            "stage": "06_sentiment_results",
            "nm_id": self.product_id,
            "n_results": len(items),
            "results": items,
        }
        _dump(data, self._path("06_sentiment_results.json"))

    # ── Стадия 7: Aggregation input ───────────────────────────────────────────

    def save_aggregation_input(self, agg_input: list) -> None:
        """agg_input: List[Dict] из _build_aggregation_input()"""
        items = [
            {
                "review_id": row["review_id"],
                "aspects": {k: round(float(v), 4) for k, v in row["aspects"].items()},
                "fraud_weight": round(float(row["fraud_weight"]), 4),
                "date": row["date"].isoformat() if hasattr(row.get("date"), "isoformat") else str(row.get("date", "")),
            }
            for row in agg_input
        ]
        data = {
            "stage": "07_aggregation_input",
            "nm_id": self.product_id,
            "n_reviews": len(items),
            "input": items,
        }
        _dump(data, self._path("07_aggregation_input.json"))

    # ── Финальный результат ───────────────────────────────────────────────────

    def save_pipeline_result(self, result: Any) -> None:
        """result: PipelineResult"""
        data = {
            "stage": "08_pipeline_result",
            "product_id": result.product_id,
            "reviews_processed": result.reviews_processed,
            "processing_time": result.processing_time,
            "aspect_keywords": result.aspect_keywords,
            "diagnostics": getattr(result, "diagnostics", {}),
            "aspects": {
                name: {k: (float(v) if isinstance(v, (float, int, np.floating)) else v)
                       for k, v in metrics.items()}
                for name, metrics in result.aspects.items()
            },
        }
        _dump(data, self._path("08_pipeline_result.json"))


# ── Deserializers (replay с любой стадии) ─────────────────────────────────────

def load_candidates_snapshot(path: str | Path) -> list:
    """
    Загружает снепшот 02_candidates.json → плоский List[Candidate].
    Replay: передать результат в scorer.score_and_select().
    """
    from src.schemas.models import Candidate
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = []
    for rev_id, cands in data["per_review"].items():
        for c in cands:
            result.append(Candidate(
                span=c["span"],
                sentence=c["sentence"],
                token_indices=tuple(c["token_indices"]),
            ))
    return result


def load_scored_snapshot(path: str | Path) -> list:
    """
    Загружает снепшот 03_scored.json → List[ScoredCandidate].
    Replay: передать результат в clusterer.cluster().
    """
    from src.schemas.models import ScoredCandidate
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = []
    for item in data["candidates"]:
        emb = np.array(item["embedding"], dtype=np.float32) if "embedding" in item else None
        result.append(ScoredCandidate(
            span=item["span"],
            score=item["score"],
            sentence=item["sentence"],
            embedding=emb,
        ))
    return result


def load_clusters_snapshot(path: str | Path) -> dict:
    """
    Загружает снепшот 04_clusters.json → Dict[str, AspectInfo].
    Replay: передать результат в _build_sentiment_pairs() пайплайна.
    """
    from src.schemas.models import AspectInfo
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for name, entry in data["aspects"].items():
        centroid = (
            np.array(entry["centroid_embedding"], dtype=np.float32)
            if "centroid_embedding" in entry
            else np.zeros(312, dtype=np.float32)
        )
        result[name] = AspectInfo(
            keywords=entry["keywords"],
            centroid_embedding=centroid,
            keyword_weights=entry.get("keyword_weights", []),
            nli_label=entry.get("nli_label", ""),
        )
    return result


def load_sentiment_results_snapshot(path: str | Path) -> list:
    """
    Загружает снепшот 06_sentiment_results.json → List[SentimentResult].
    Replay: передать результат в _build_aggregation_input() пайплайна.
    """
    from src.schemas.models import SentimentResult
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SentimentResult(
            review_id=r["review_id"],
            aspect=r["aspect"],
            sentence=r["sentence"],
            score=r["score"],
            p_ent_pos=r["p_ent_pos"],
            p_ent_neg=r["p_ent_neg"],
            confidence=r.get("confidence", 1.0),
        )
        for r in data["results"]
    ]


def list_snapshots(snapshot_dir: str | Path, product_id: int) -> List[Path]:
    """Возвращает все снепшот-файлы для одного продукта в хронологическом порядке."""
    d = Path(snapshot_dir) / f"nm{product_id}"
    if not d.exists():
        return []
    return sorted(d.glob("*.json*"))
