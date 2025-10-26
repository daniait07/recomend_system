from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging

from additional.models import Entity, NERProvider, EmotionLabels

# --- Опциональные зависимости ---
try:
    from huggingface_hub import InferenceApi
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    import torch
    from transformers import pipeline, Pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceNERError(RuntimeError):
    """Ошибка вызова Hugging Face API или локальной модели."""


class HuggingFaceNER(NERProvider):
    """
    Провайдер NER через Hugging Face:
    - Предпочтительно использует InferenceApi (HF Inference endpoint)
    - При отсутствии токена/интернета может использовать локальный pipeline (transformers)
    """

    def __init__(
        self,
        model: str,
        hf_token: Optional[str] = None,
        use_local_fallback: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self._model = model
        self._token = hf_token
        self._use_local = use_local_fallback
        self._logger = logger or logging.getLogger(self.__class__.__name__)

        # Попробуем создать InferenceApi
        self._inference = None
        if HF_AVAILABLE and self._token:
            try:
                self._inference = InferenceApi(repo_id=self._model, token=self._token)
            except Exception as exc:
                self._logger.warning("Could not initialize HF InferenceApi: %s", exc)
                self._inference = None

        # Подготовим локальный pipeline при необходимости
        self._local_pipeline: Optional[Pipeline] = None
        if self._use_local:
            if not TRANSFORMERS_AVAILABLE:
                raise HuggingFaceNERError("Local fallback requested but transformers/torch not available")
            try:
                self._local_pipeline = pipeline("ner", model=self._model, grouped_entities=True)
            except Exception as exc:
                self._logger.exception("Failed to create local transformers pipeline: %s", exc)
                raise HuggingFaceNERError("Failed to initialize local transformers pipeline") from exc

        if not self._inference and not self._local_pipeline:
            self._logger.warning(
                "HuggingFaceNER created without inference client nor local pipeline. Will fail on extract()."
            )

    def get_name(self) -> str:
        mode = "inference" if self._inference else ("local" if self._local_pipeline else "none")
        return f"HuggingFace [{self._model}] ({mode})"

    def extract(self, text: str) -> List[Entity]:
        if not text or not text.strip():
            return []

        # 1) Попытка через InferenceApi
        if self._inference:
            try:
                resp = self._inference(inputs=text)
                entities = self._normalize_inference_response(resp)
                return [Entity(text=e["word"], label=self._heuristic_label(e["word"])) for e in entities]
            except Exception as exc:
                self._logger.exception("HF InferenceApi error: %s", exc)
                # fallthrough к локальному пайплайну

        # 2) Локальный transformers pipeline
        if self._local_pipeline:
            try:
                ner_out = self._local_pipeline(text)
                normalized = []
                for item in ner_out:
                    word = item.get("word") or item.get("entity_group") or ""
                    normalized.append({
                        "word": word,
                        "score": item.get("score", 0.0),
                        "entity": item.get("entity_group", "")
                    })
                return [Entity(text=item["word"], label=self._heuristic_label(item["word"])) for item in normalized]
            except Exception as exc:
                self._logger.exception("Local transformers pipeline error: %s", exc)
                raise HuggingFaceNERError("Both HF Inference and local pipeline failed") from exc

        raise HuggingFaceNERError("No available HF inference method (no token and no local pipeline).")

    # --- Вспомогательные методы ---
    @staticmethod
    def _normalize_inference_response(resp: Any) -> List[Dict[str, Any]]:
        """
        Нормализует разные формы ответа InferenceApi в список {'word': str, ...}
        """
        if resp is None:
            return []
        if isinstance(resp, list):
            normalized = []
            for item in resp:
                if isinstance(item, dict):
                    word = item.get("word") or item.get("entity_group") or item.get("entity") or ""
                    normalized.append({"word": word, **item})
                else:
                    normalized.append({"word": str(item)})
            return normalized
        if isinstance(resp, dict):
            if "entities" in resp and isinstance(resp["entities"], list):
                return HuggingFaceNER._normalize_inference_response(resp["entities"])
            word = resp.get("word") or resp.get("entity_group") or resp.get("entity") or ""
            return [{"word": word, **resp}]
        return [{"word": str(resp)}]

    @staticmethod
    def _heuristic_label(text: str) -> str:
        """
        Простая эвристика: классифицирует текст на POSITIVE/NEGATIVE/USER_EXPERIENCE/TARGET
        """
        txt = text.lower()

        positive = {"хороший", "отличный", "удобный", "доволен", "нравится", "люблю"}
        negative = {"плохой", "шумный", "дорогой", "разочарован", "не нравится", "ненавижу"}
        experience = {"купил", "тестировал", "опыт", "использую", "пользуюсь", "приобрел", "приобрела"}

        if any(k in txt for k in positive):
            return EmotionLabels.POSITIVE_FEATURE
        if any(k in txt for k in negative):
            return EmotionLabels.NEGATIVE_FEATURE
        if any(k in txt for k in experience):
            return EmotionLabels.USER_EXPERIENCE
        return EmotionLabels.TARGET
