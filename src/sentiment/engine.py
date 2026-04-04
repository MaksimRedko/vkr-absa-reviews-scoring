"""
Модуль NLI Sentiment (v3 — dual hypothesis)

Определяет полярность (позитив/негатив) каждого аспекта в каждом предложении
через Natural Language Inference с двумя встречными гипотезами.

Две гипотезы для каждой пары (sentence, aspect):
  H_pos = "Автор доволен {aspect}"
  H_neg = "Автор недоволен {aspect}"

Формула:
  Score = 1 + 4 · P_ent_pos / (P_ent_pos + P_ent_neg + ε)

Это устраняет однонаправленный bias модели: если модель склонна давать
высокий entailment для любой гипотезы, это сокращается в числителе и знаменателе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.configs import config
from src.stages import SentimentStage

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


@dataclass
class SentimentResult:
    """Результат анализа сентимента для одной пары (предложение, аспект)"""
    review_id: str
    aspect: str
    sentence: str
    score: float          # [1.0, 5.0]
    p_ent_pos: float      # P(entailment | H_pos)
    p_ent_neg: float      # P(entailment | H_neg)
    confidence: float = 1.0  # вес из soft-anchor / span
class SentimentEngine(SentimentStage):
    """
    NLI-based sentiment engine v3 (dual hypothesis).

    Два forward pass на батч: один для H_pos, один для H_neg.
    Score = 1 + 4 · P_ent_pos / (P_ent_pos + P_ent_neg + ε)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.models.nli_path,
            local_files_only=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.models.nli_path,
            local_files_only=True,
        )
        self.model.eval()

        self.h_pos: str = config.sentiment.hypothesis_template_pos
        self.h_neg: str = config.sentiment.hypothesis_template_neg
        self.batch_size: int = config.sentiment.batch_size
        self.epsilon: float = config.sentiment.score_epsilon

        self.num_labels = self.model.config.num_labels
        self.id2label = self.model.config.id2label

        self.ent_idx = 0
        for idx, label in self.id2label.items():
            if label == "entailment":
                self.ent_idx = int(idx)

        print(f"[SentimentEngine v3] dual-hypothesis, {self.num_labels} classes, "
              f"ent_idx={self.ent_idx}, device={self.device}")

    def batch_analyze(
        self,
        pairs: List[
            Union[
                Tuple[str, str, str],
                Tuple[str, str, str, float],
                Tuple[str, str, str, str, float],
            ]
        ],
    ) -> List[SentimentResult]:
        if not pairs:
            return []

        results = []
        n_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        batch_indices = range(0, len(pairs), self.batch_size)
        for i in tqdm(
            batch_indices,
            desc="      NLI батчи",
            total=n_batches,
            leave=False,
            unit="batch",
        ):
            batch = pairs[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results

    def _infer_entailment(
        self, premises: List[str], hypotheses: List[str]
    ) -> np.ndarray:
        """Возвращает P(entailment) для каждой пары premise-hypothesis."""
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs[:, self.ent_idx]

    def _process_batch(
        self,
        batch: List[
            Union[
                Tuple[str, str, str],
                Tuple[str, str, str, float],
                Tuple[str, str, str, str, float],
            ]
        ],
    ) -> List[SentimentResult]:
        review_ids = [p[0] for p in batch]
        premises = [p[1] for p in batch]
        aspects_tag = [p[2] for p in batch]

        nli_for_hyp: List[str] = []
        confidences: List[float] = []
        for p in batch:
            if len(p) >= 5:
                nli_for_hyp.append(p[3])
                confidences.append(float(p[4]))
            elif len(p) == 4:
                nli_for_hyp.append(p[2])
                confidences.append(float(p[3]))
            else:
                nli_for_hyp.append(p[2])
                confidences.append(1.0)

        hyp_pos = [self.h_pos.format(aspect=a) for a in nli_for_hyp]
        hyp_neg = [self.h_neg.format(aspect=a) for a in nli_for_hyp]

        p_ent_pos = self._infer_entailment(premises, hyp_pos)
        p_ent_neg = self._infer_entailment(premises, hyp_neg)

        results = []
        for idx, (review_id, sentence, aspect_orig) in enumerate(
            zip(review_ids, premises, aspects_tag)
        ):
            pp = float(p_ent_pos[idx])
            pn = float(p_ent_neg[idx])

            score = 1.0 + 4.0 * pp / (pp + pn + self.epsilon)
            score = max(1.0, min(5.0, score))

            results.append(SentimentResult(
                review_id=review_id,
                aspect=aspect_orig,
                sentence=sentence,
                score=score,
                p_ent_pos=pp,
                p_ent_neg=pn,
                confidence=confidences[idx],
            ))

        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Добавить корень проекта в путь для импортов
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.stdout.reconfigure(encoding="utf-8")
    
    from configs.configs import config  # noqa: E402
    
    # Тестовые пары
    test_pairs = [
        ("r1", "Экран шикарный, яркие цвета", "Качество экрана"),
        ("r2", "Сдохла батарея за день", "Батарея"),
        ("r3", "Ну такое, средненько", "Общее впечатление"),
        ("r4", "Коробка мятая, но сам товар целый", "Логистика"),
        ("r5", "Быстрая доставка, всё пришло за 2 дня", "Логистика"),
        ("r6", "Качество сборки на высоте, материал отличный", "Качество"),
    ]
    
    print("Инициализация SentimentEngine...")
    engine = SentimentEngine()
    
    print(f"\nТест на {len(test_pairs)} парах:")
    print("=" * 80)
    
    results = engine.batch_analyze(test_pairs)
    
    for res in results:
        print(f"\nReview: {res.review_id}")
        print(f"Аспект: {res.aspect}")
        print(f"Предложение: {res.sentence}")
        print(f"Score: {res.score:.2f}")
        print(f"  P_ent_pos={res.p_ent_pos:.3f}, P_ent_neg={res.p_ent_neg:.3f}")
    
    print("\n" + "=" * 80)
    print("Тест завершён")
