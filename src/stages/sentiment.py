"""
Модуль NLI Sentiment (v4 — одна гипотеза, entailment vs contradiction)

Полярность (позитив/негатив) через одну гипотезу на пару (sentence, aspect):
  H = "{aspect} — это хорошо"  (шаблон из config.sentiment.hypothesis_template_pos)

Из одного трёхклассового прогона берём P(entailment) и P(contradiction); нейтраль
остаётся в полном softmax, но в скоре не участвует:

  Score = 1 + 4 · P_ent / (P_ent + P_contra + ε), затем clamp [1, 5]

В SentimentResult: p_ent_pos = P(entailment), p_ent_neg = P(contradiction)
(имена полей сохранены для совместимости с фильтром релевантности и снепшотами).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.configs import config
from src.schemas.models import SentimentResult
from src.stages.contracts import SentimentStage

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


class SentimentEngine(SentimentStage):
    """
    NLI-based sentiment engine v4 (single hypothesis).

    Один forward на батч пар (premise, hypothesis).
    Score = 1 + 4 · P_ent / (P_ent + P_contra + ε)
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

        self.h_template: str = config.sentiment.hypothesis_template_pos
        self.batch_size: int = config.sentiment.batch_size
        self.epsilon: float = config.sentiment.score_epsilon
        self.temperature: float = float(config.sentiment.temperature)

        self.num_labels = self.model.config.num_labels
        self.id2label = self.model.config.id2label

        self.ent_idx = 0
        self.contra_idx = 0
        for idx, label in self.id2label.items():
            lab = str(label).lower()
            if lab == "entailment":
                self.ent_idx = int(idx)
            if lab == "contradiction":
                self.contra_idx = int(idx)

        print(
            f"[SentimentEngine v4] single-hypothesis, {self.num_labels} classes, "
            f"ent_idx={self.ent_idx}, contra_idx={self.contra_idx}, device={self.device}"
        )

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

    def _forward_logits_tensor(
        self, premises: List[str], hypotheses: List[str]
    ) -> torch.Tensor:
        """Сырые логиты (B, num_labels) на device."""
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            return self.model(**inputs).logits

    def batch_collect_logits(
        self,
        pairs: List[
            Union[
                Tuple[str, str, str],
                Tuple[str, str, str, float],
                Tuple[str, str, str, str, float],
            ]
        ],
    ) -> np.ndarray:
        """
        Логиты одного прогона на пару: (N, num_labels), порядок строк = порядок pairs.
        """
        if not pairs:
            return np.zeros((0, self.num_labels), dtype=np.float32)

        blocks: List[np.ndarray] = []
        batch_indices = range(0, len(pairs), self.batch_size)
        for i in tqdm(
            batch_indices,
            desc="      NLI logits",
            total=(len(pairs) + self.batch_size - 1) // self.batch_size,
            leave=False,
            unit="batch",
        ):
            batch = pairs[i : i + self.batch_size]
            premises = [p[1] for p in batch]
            hyp_aspects = self._hypothesis_aspects(batch)
            hyp_texts = [self.h_template.format(aspect=a) for a in hyp_aspects]
            logits = self._forward_logits_tensor(premises, hyp_texts)
            blocks.append(logits.cpu().numpy())

        return np.vstack(blocks)

    @staticmethod
    def _hypothesis_aspects(
        batch: List[
            Union[
                Tuple[str, str, str],
                Tuple[str, str, str, float],
                Tuple[str, str, str, str, float],
            ]
        ],
    ) -> List[str]:
        out: List[str] = []
        for p in batch:
            if len(p) >= 5:
                out.append(p[3])
            elif len(p) == 4:
                out.append(p[2])
            else:
                out.append(p[2])
        return out

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

        nli_for_hyp = self._hypothesis_aspects(batch)
        confidences: List[float] = []
        for p in batch:
            if len(p) >= 5:
                confidences.append(float(p[4]))
            elif len(p) == 4:
                confidences.append(float(p[3]))
            else:
                confidences.append(1.0)

        hyp_texts = [self.h_template.format(aspect=a) for a in nli_for_hyp]
        logits = self._forward_logits_tensor(premises, hyp_texts)
        probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
        p_ent = probs[:, self.ent_idx]
        p_contra = probs[:, self.contra_idx]

        results = []
        for idx, (review_id, sentence, aspect_orig) in enumerate(
            zip(review_ids, premises, aspects_tag)
        ):
            pe = float(p_ent[idx])
            pc = float(p_contra[idx])
            denom = pe + pc + self.epsilon
            score = 1.0 + 4.0 * pe / denom
            score = max(1.0, min(5.0, score))

            results.append(
                SentimentResult(
                    review_id=review_id,
                    aspect=aspect_orig,
                    sentence=sentence,
                    score=score,
                    p_ent_pos=pe,
                    p_ent_neg=pc,
                    confidence=confidences[idx],
                )
            )

        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.stdout.reconfigure(encoding="utf-8")

    from configs.configs import config  # noqa: E402

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
        print(f"  P(ent)={res.p_ent_pos:.3f}, P(contra)={res.p_ent_neg:.3f}")

    print("\n" + "=" * 80)
    print("Тест завершён")
