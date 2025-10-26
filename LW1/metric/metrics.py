from __future__ import annotations
import json
import os
import time
import logging
from typing import Any
from datetime import datetime
from additional.settings import settings

logger = logging.getLogger("metrics")


class MetricsLogger:
    """
    Простой JSONL логгер метрик, сохраняет каждую запись в отдельную строку.
    Удобен для анализа в pandas, Excel, BigQuery и т.д.
    """

    def __init__(self, filepath: str | None = None):
        self._filepath = filepath or settings.metrics.file_path
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)

    def log(
        self,
        model_name: str,
        input_text: str,
        output_entities: list[dict[str, Any]],
        latency_sec: float,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "input_length": len(input_text),
            "entities_count": len(output_entities),
            "latency_sec": round(latency_sec, 3),
            "success": success,
            "error": error,
            "input_preview": input_text[:100],
            "entities": output_entities,
        }

        try:
            with open(self._filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.exception("Failed to write metrics: %s", exc)

    def measure_and_log(self, provider, text: str) -> list:
        """
        Измеряет время выполнения extract(), сохраняет метрики, возвращает результат.
        """
        start = time.perf_counter()
        try:
            result = provider.extract(text)
            latency = time.perf_counter() - start
            entities_dicts = [e.__dict__ for e in result]
            self.log(provider.get_name(), text, entities_dicts, latency, success=True)
            return result
        except Exception as exc:
            latency = time.perf_counter() - start
            self.log(provider.get_name(), text, [], latency, success=False, error=str(exc))
            raise

