from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Импортируем наш сервис
from app.services.analyzer import ProductAnalyzerService
from app.core.data_loader import DataLoader

# Глобальная переменная для сервиса (синглтон)
service: ProductAnalyzerService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Событие запуска приложения.
    Здесь мы загружаем тяжелые модели в память (Cold Start).
    """
    global service
    try:
        print("🌍 ЗАПУСК API: Инициализация нейросетей...")
        # Указываем путь к БД (убедись, что файл лежит в корне или укажи полный путь)
        service = ProductAnalyzerService(db_path="dataset.db")
        print("✅ API готов к приему запросов!")
    except Exception as e:
        print(f"❌ Критическая ошибка запуска: {e}")

    yield

    print("🛑 Остановка API...")
    # Тут можно очистить память, если нужно


app = FastAPI(title="Aspect Sentiment API", lifespan=lifespan)

# Разрешаем запросы с фронтенда (Streamlit обычно висит на 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Схемы данных для Swagger UI ---
class AnalyzeRequest(BaseModel):
    nm_id: int
    limit: int = 50


# --- Эндпоинты ---

@app.get("/products/top")
async def get_top_products():
    """Возвращает список популярных товаров для выбора в UI"""
    try:
        # Используем лоадер напрямую для быстрого списка
        df = service.loader.get_top_products(limit=10)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_product(request: AnalyzeRequest):
    """Запускает тяжелый ML-пайплайн для товара"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = service.analyze_product(request.nm_id, limit=request.limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": service is not None}