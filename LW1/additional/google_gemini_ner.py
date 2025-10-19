from typing import List
import json
from pydantic import BaseModel, Field
from google import genai
from additional.models import Entity, NERProvider

class _GeminiEntity(BaseModel):
    text: str = Field(..., description="Text of the entity")
    label: str = Field(..., description="Emotional category: TARGET, POSITIVE_FEATURE, NEGATIVE_FEATURE, USER_EXPERIENCE")

class _GeminiOutput(BaseModel):
    entities: List[_GeminiEntity]

_GEMINI_PROMPT_TEMPLATE = """

Извлекает из текста ключевые сущности и для каждой уазывает категорию
- TARGET — объект, о котором идёт речь (товар, бренд, услуга)
- POSITIVE_FEATURE — положительная характеристика (что нравится)
- NEGATIVE_FEATURE — отрицательная характеристика (что не нравится)
- USER_EXPERIENCE — действие или эмоция пользователя (купил, доволен, разочарован)

Текст для анализа:
\"\"\"{text}\"\"\"

Верни строго JSON в формате:
{{
  "entities": [
    {{"text": "...", "label": "TARGET"}},
    {{"text": "...", "label": "POSITIVE_FEATURE"}},
    ...
  ]
}}
 
"""
 
class GoogleGeminiNER(NERProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def get_name(self) -> str:
        return f"Google Gemini {self._model}"

    def extract(self, text: str) -> List[Entity]:
        prompt = _GEMINI_PROMPT_TEMPLATE.format(text=text)

        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _GeminiOutput,
            },
        )

        parsed = resp.parsed
        if parsed is None:
            data = json.loads(resp.text)
            entities = data.get("entities", [])
            return [
                Entity(text=entity["text"], label=entity["label"])
                for entity in entities
                if "text" in entity and "label" in entity
            ]
        return [Entity(text=ent.text, label=ent.label) for ent in parsed.entities]
