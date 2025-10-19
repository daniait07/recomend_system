from typing import List
from pydantic import BaseModel, Field
from google import genai
from providers.models import Entity, NERProvider
import json


class _GeminiEntity(BaseModel):
    text: str = Field(..., description="Text")
    label: str = Field(..., description="Entity type (e.g., PERSON, ORG, LOCATION)")


class _GeminiOutput(BaseModel):
    entities: List[_GeminiEntity]


_GEMINI_PROMPT_TEMPLATE = """You are a NER (Named Entity Recognition) model.
Extract named entities from the following text:

\"\"\"{text}\"\"\"

Return strictly JSON with the schema:
{{
  "entities": [
    {{"text": string, "label": string}},
  ]
}}

Guidelines:
- Prefer coarse labels: PERSON, ORG, LOCATION, DATE, EVENT, PRODUCT, WORK, OTHER.
- DO NOT add extra fields. DO NOT add explanations. Output JSON only.
"""


class GoogleGeminiNER(NERProvider):
    def __init__(
        self, api_key: str, model: str = "gemini-2.5-flash"
    ):
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

        return [Entity(text=entinty.text, label=entinty.label) for entinty in parsed.entities]
