from typing import List, Dict, Any
import requests
from additional.models import Entity, NERProvider, EmotionLabels
 
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
        headers = {"Authorization": f"Token {self._api_key}"}
        payload: Dict[str, Any] = {"text": text}

        # HTTP-запрос
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
            typ = entity.get("type", "").upper()   
 
            label = EmotionLabels.TARGET  

            positive_keywords = {"хороший", "отличный", "удобный", "доволен", "нравится"}
            negative_keywords = {"плохой", "шумный", "дорогой", "разочарован", "не нравится"}
            user_experience_keywords = {"купил", "тестировал", "опыт", "использую"}

            txt_lower = txt.lower() if txt else ""

            if any(word in txt_lower for word in positive_keywords):
                label = EmotionLabels.POSITIVE_FEATURE
            elif any(word in txt_lower for word in negative_keywords):
                label = EmotionLabels.NEGATIVE_FEATURE
            elif any(word in txt_lower for word in user_experience_keywords):
                label = EmotionLabels.USER_EXPERIENCE

            out.append(Entity(text=str(txt), label=label))

        return out
