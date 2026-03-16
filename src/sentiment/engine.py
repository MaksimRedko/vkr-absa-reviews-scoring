"""
Модуль NLI Sentiment (v2)

Определяет полярность (позитив/негатив) каждого аспекта в каждом предложении
через Natural Language Inference.

Ключевые отличия от v1:
- Убран этап Presence (его роль выполняет KeyBERT из Discovery)
- Одна гипотеза вместо двух: "Автор доволен {aspect}"
- Батчевый инференс вместо поштучного
- Работа с целым предложением, без скользящего окна
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.configs import config


@dataclass
class SentimentResult:
    """Результат анализа сентимента для одной пары (предложение, аспект)"""
    review_id: str
    aspect: str
    sentence: str
    score: float  # [1.0, 5.0]
    p_entailment: float
    p_contradiction: float
    p_neutral: float


class SentimentEngine:
    """
    NLI-based sentiment engine v2.
    
    Использует одну гипотезу: "Автор доволен {aspect}".
    
    Формула конвертации в скор [1, 5]:
        Score = 1 + 4 · P(entailment)
    
    Интуиция:
        - P(ent) ≈ 1.0 → Score ≈ 5 (автор явно доволен)
        - P(ent) ≈ 0.5 → Score ≈ 3 (нейтрально)
        - P(ent) ≈ 0.0 → Score ≈ 1 (автор недоволен)
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
        
        self.hypothesis_template: str = config.sentiment.hypothesis_template
        self.batch_size: int = config.sentiment.batch_size
        self.epsilon: float = config.sentiment.score_epsilon
        
        # Модель rubert-base-cased-nli-twoway возвращает 2 класса: entailment, not_entailment
        self.num_labels = 2
    
    def batch_analyze(
        self, pairs: List[Tuple[str, str, str]]
    ) -> List[SentimentResult]:
        """
        Батчевый анализ пар (review_id, sentence, aspect).
        
        Args:
            pairs: Список (review_id, sentence, aspect_name)
        
        Returns:
            Список SentimentResult с оценками
        """
        if not pairs:
            return []
        
        results = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(
        self, batch: List[Tuple[str, str, str]]
    ) -> List[SentimentResult]:
        """Обработка одного батча пар"""
        review_ids = [pair[0] for pair in batch]
        premises = [pair[1] for pair in batch]
        aspects = [pair[2] for pair in batch]
        
        # Формирование гипотез
        hypotheses = [
            self.hypothesis_template.format(aspect=aspect) for aspect in aspects
        ]
        
        # Токенизация
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Конвертация в результаты
        results = []
        for idx, (review_id, sentence, aspect) in enumerate(
            zip(review_ids, premises, aspects)
        ):
            p_ent = float(probs[idx][0])
            p_not_ent = float(probs[idx][1])
            
            # Простая линейная конверсия: Score = 1 + 4 · P(ent)
            score = 1.0 + 4.0 * p_ent
            score = max(1.0, min(5.0, score))
            
            results.append(
                SentimentResult(
                    review_id=review_id,
                    aspect=aspect,
                    sentence=sentence,
                    score=score,
                    p_entailment=p_ent,
                    p_contradiction=p_not_ent,
                    p_neutral=0.0,
                )
            )
        
        return results
    
    def _convert_to_score(self, p_ent: float) -> float:
        """
        Конвертация NLI-вероятности в скор [1, 5].
        
        Формула: Score = 1 + 4 · P(entailment)
        """
        score = 1.0 + 4.0 * p_ent
        return max(1.0, min(5.0, score))


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
        print(f"  P(ent)={res.p_entailment:.3f}, P(contr)={res.p_contradiction:.3f}, P(neut)={res.p_neutral:.3f}")
    
    print("\n" + "=" * 80)
    print("Тест завершён")
