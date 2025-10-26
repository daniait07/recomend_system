from __future__ import annotations
from typing import List, Protocol
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """
    Представление распознанной сущности.
    text: исходный фрагмент текста
    label: назначенная метка (например, POSITIVE_FEATURE)
    """
    text: str
    label: str


class EmotionLabels:
    """Набор возможных меток для анализа аспектов/эмоций."""
    TARGET = "TARGET"
    POSITIVE_FEATURE = "POSITIVE_FEATURE"
    NEGATIVE_FEATURE = "NEGATIVE_FEATURE"
    USER_EXPERIENCE = "USER_EXPERIENCE"


class NERProvider(Protocol):
    """
    Протокол / интерфейс для провайдеров NER.
    Любой провайдер должен реализовать get_name() и extract(text).
    Используем Protocol, чтобы было легко мокать в тестах.
    """

    def get_name(self) -> str:
        ...

    def extract(self, text: str) -> List[Entity]:
        ...
