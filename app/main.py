"""
FastAPI обёртка ABSA Pipeline v2.

Эндпоинты:
  GET  /products/top  — список популярных товаров
  POST /analyze       — запуск полного ML-пайплайна
  GET  /health        — проверка здоровья
"""

from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import ABSAPipeline


pipeline: Optional[ABSAPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        print("[API] Инициализация пайплайна...")
        pipeline = ABSAPipeline(db_path="data/dataset.db")
        print("[API] Готов к приёму запросов")
    except Exception as e:
        print(f"[API] Критическая ошибка запуска: {e}")
    yield
    print("[API] Остановка")


app = FastAPI(title="Aspecta AI — ABSA Pipeline v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    nm_id: int
    limit: int = 100


class PersonalRatingRequest(BaseModel):
    nm_id: int
    weights: Dict[str, float]


# ------------------------------------------------------------------
# Эндпоинты
# ------------------------------------------------------------------

@app.get("/products/top")
async def get_top_products():
    """Список популярных товаров для UI."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        df = pipeline.loader.get_top_products(limit=10)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_product(request: AnalyzeRequest):
    """Запуск полного ML-пайплайна для товара."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = pipeline.analyze_product(request.nm_id, limit=request.limit)

        return {
            "product_id": result.product_id,
            "reviews_processed": result.reviews_processed,
            "processing_time": result.processing_time,
            "aspects": result.aspects,
            "aspect_keywords": result.aspect_keywords,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": pipeline is not None}
