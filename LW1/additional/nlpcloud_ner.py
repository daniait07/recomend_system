from __future__ import annotations
from typing import List, Dict, Any
import logging
import requests

from additional.models import Entity, NERProvider, EmotionLabels


class NLPCloudEntitiesError(RuntimeError):
    """Ошибка вызова или обработки ответа от NLP Cloud."""


class NLPCloudNER(NERProvider):
    """
    Провайдер NER через NLP Cloud API.
    - Надёжная обработка ошибок
    - Валидация входа/выхода
    - Простейшая эвристическая категоризация (можно заменить на ML-метод)
    """

    DEFAULT_TIMEOUT = 15.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "en_core_web_lg",
        timeout: float | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")

        self._url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = float(timeout) if timeout is not None else self.DEFAULT_TIMEOUT
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def get_name(self) -> str:
        return f"NLP Cloud [{self._model}]"

    def extract(self, text: str) -> List[Entity]:
        """
        Отправляет запрос к /{model}/entities и возвращает список Entity.
        Возвращает [] для пустого/пустого строки.
        """
        if not text or not text.strip():
            return []

        endpoint = f"{self._url}/{self._model}/entities"
        headers = {"Authorization": f"Token {self._api_key}"}
        payload: Dict[str, Any] = {"text": text}

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            self._logger.exception("HTTP error calling NLP Cloud: %s", exc)
            raise NLPCloudEntitiesError("HTTP error when calling NLP Cloud") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            self._logger.exception("Invalid JSON from NLP Cloud: %s", exc)
            raise NLPCloudEntitiesError("Invalid JSON response from NLP Cloud") from exc

        entities = data.get("entities")
        if not isinstance(entities, list):
            self._logger.error("Unexpected response shape from NLP Cloud: %s", data)
            raise NLPCloudEntitiesError("Missing or invalid 'entities' list in response")

        results: List[Entity] = []
        for raw in entities:
            if not isinstance(raw, dict):
                continue
            txt = raw.get("text")
            if not txt or not isinstance(txt, str):
                continue

            label = self._heuristic_label(txt, raw)
            results.append(Entity(text=txt.strip(), label=label))

        return results

    # --- Вспомогательные ---
    @staticmethod
    def _heuristic_label(text: str, raw_entity: Dict[str, Any] | None = None) -> str:
        """
        Простейшая эвристика, выделяющая позитив/негатив/опыт/целевой объект.
        Можно заменить на ML/сервисную классификацию.
        """
        txt = text.lower()

        positive_keywords = {"хороший", "отличный", "удобный", "доволен", "нравится", "люблю"}
        negative_keywords = {"плохой", "шумный", "дорогой", "разочарован", "не нравится", "ненавижу"}
        experience_keywords = {"купил", "тестировал", "опыт", "использую", "пользуюсь", "приобрел", "приобрела"}

        if any(k in txt for k in positive_keywords):
            return EmotionLabels.POSITIVE_FEATURE
        if any(k in txt for k in negative_keywords):
            return EmotionLabels.NEGATIVE_FEATURE
        if any(k in txt for k in experience_keywords):
            return EmotionLabels.USER_EXPERIENCE
        return EmotionLabels.TARGET

