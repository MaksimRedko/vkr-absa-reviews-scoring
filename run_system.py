from app.services.analyzer import ProductAnalyzerService
from app.core.data_loader import DataLoader
import json


def main():
    # 1. Ищем ID популярного товара
    loader = DataLoader("data/dataset.db")
    top_products = loader.get_top_products(limit=1)

    if top_products.empty:
        print("❌ База пуста!")
        return

    target_nm_id = int(top_products.iloc[0]['nm_id'])
    print(f"🎯 Выбран топ-товар ID: {target_nm_id} (Отзывов: {top_products.iloc[0]['review_count']})")

    # 2. Инициализируем сервис
    service = ProductAnalyzerService("data/dataset.db")

    # 3. Запускаем анализ (берем только 30 отзывов для теста, чтобы быстро)
    # Потом увеличишь limit до 100-200
    result = service.analyze_product(target_nm_id, limit=30)

    # 4. Вывод результата
    print("\n" + "=" * 50)
    print(f"✅ АНАЛИЗ ЗАВЕРШЕН за {result['processing_time']} сек.")
    print("=" * 50)

    for aspect, data in result['aspects'].items():
        score = data['score']
        # Рисуем прогресс-бар в консоли
        bars = int(score * 2)
        visual = "🟦" * bars + "⬜" * (10 - bars)

        print(f"{visual} {aspect}: {score} (raw: {data['raw_mean']}, споры: {data['controversy']})")

    # Сохраним в JSON для проверки
    with open("result_dump.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("\n📁 Полный отчет сохранен в result_dump.json")


if __name__ == "__main__":
    main()