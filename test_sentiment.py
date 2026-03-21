import sys
sys.stdout.reconfigure(encoding="utf-8")

from src.sentiment.engine import SentimentEngine

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
