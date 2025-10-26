from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    text: str
    label: str


class NERProvider(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        raise NotImplementedError
