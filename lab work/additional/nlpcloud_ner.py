from typing import List, Dict, Any
import requests

from providers.models import Entity, NERProvider


class NLPCloudEntitiesError(RuntimeError):
    pass


class NLP_CLOUD_NER(NERProvider):
    def __init__(self, url: str, api_key: str, model: str = "en_core_web_lg", timeout: float = 15.0):
        self._url = url
        self._api_key = api_key
        self._model = model
        self._timeout = float(timeout)

    def get_name(self) -> str:
        return f"NLP Cloud [{self._model}]"

    def extract(self, text: str) -> List[Entity]:
        url = f"{self._url}/{self._model}/entities"
        headers = {
            "Authorization": f"Token {self._api_key}",
        }
        payload: Dict[str, Any] = {"text": text}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise NLPCloudEntitiesError(f"HTTP error calling NLP Cloud Entities: {e}") from e

        if "entities" not in data or not isinstance(data["entities"], list):
            raise NLPCloudEntitiesError("Unexpected response: no 'entities' list")

        out: List[Entity] = []
        for entity in data["entities"]:
            txt = entity.get("text")
            typ = entity.get("type")
            out.append(Entity(text=str(txt), label=str(typ).upper()))
        return out
