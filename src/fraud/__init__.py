"""Модуль AntiFraud — расчёт весов доверия для отзывов"""

from .engine import AntiFraudEngine, TrustResult

__all__ = ["AntiFraudEngine", "TrustResult"]
