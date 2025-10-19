from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    text: str
    label: str

class EmotionLabels:
    TARGET = "TARGET"
    POSITIVE_FEATURE = "POSITIVE_FEATURE"
    NEGATIVE_FEATURE = "NEGATIVE_FEATURE"
    USER_EXPERIENCE = "USER_EXPERIENCE"

class NERProvider(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        raise NotImplementedError
