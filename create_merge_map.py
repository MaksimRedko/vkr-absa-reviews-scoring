import json

# Mapping based on failure analysis data
# Key: Predicted aspect name (mapped_true in candidate_assignment_debug.csv)
# Value: Compact aspect name

# Note: names must match exactly what is in 'predicted_aspect_mapped' column
# We use the raw Russian names from the audit output

merge_map = {
    # Service family
    "Обслуживание": "Service",
    "Персонал": "Service",
    "Сотрудники": "Service",
    "Сервис": "Service",
    "Отношение": "Service",
    
    # Quality family
    "Качество": "Quality",
    "Надежность": "Quality",
    "Соответствие": "Quality",
    "Брак": "Quality",
    "Дефекты": "Quality",
    
    # Appearance
    "Внешний вид": "Appearance",
    "Дизайн": "Appearance",
    "Цвет": "Appearance",
    "Модель": "Appearance",
    "Стиль": "Appearance",
    
    # Comfort
    "Комфорт": "Comfort",
    "Удобство": "Comfort",
    "Эргономика": "Comfort",
    "Практичность": "Comfort",
    
    # Price
    "Цена": "Price",
    "Стоимость": "Price",
    "Выгода": "Price",
    "Доступность": "Price",
    
    # Logistics / Packaging
    "Упаковка": "Logistics",
    "Доставка": "Logistics",
    "Срок": "Logistics",
    "Комплектация": "Logistics",
    
    # Condition
    "Состояние": "Condition",
    "Чистота": "Condition",
    "Запах": "Condition",
    "Свежесть": "Condition",
    
    # Facilities
    "Номера": "Facilities",
    "Бассейн": "Facilities",
    "СПА": "Facilities",
    "Инфраструктура": "Facilities",
    "Территория": "Facilities",
    "Парковка": "Facilities"
}

with open("aspect_merge_map.json", "w", encoding="utf-8") as f:
    json.dump(merge_map, f, ensure_ascii=False, indent=2)
