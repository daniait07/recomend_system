import json
from typing import Set, Tuple, List
from additional.models import Entity

def parse_ground_truth(raw: str) -> List[Entity]:
    """
    Поддерживает формат: {"entities":[{"text":"...","label":"POSITIVE_FEATURE"}, ...]}
    """
    data = json.loads(raw.strip())
    ents = []
    if isinstance(data, dict) and "entities" in data and isinstance(data["entities"], list):
        for entity in data["entities"]:
            if isinstance(entity, dict) and "text" in entity and "label" in entity:
                ents.append(Entity(text=str(entity["text"]), label=str(entity["label"]).upper()))
    return ents


def calculate_emotion_metrics(entities: List[Entity]) -> Dict[str, float]:
    """
    Рассчитывает эмоциональные метрики:
    - positive_count
    - negative_count
    - balance (от -1 до +1)
    - dominant_sentiment
    """

    positive_labels = {"POSITIVE_FEATURE", "POSITIVE", "GOOD", "HAPPY"}
    negative_labels = {"NEGATIVE_FEATURE", "NEGATIVE", "BAD", "SAD"}

    pos = sum(1 for e in entities if e.label in positive_labels)
    neg = sum(1 for e in entities if e.label in negative_labels)

    total = pos + neg
    if total == 0:
        balance = 0.0
        dominant = "neutral"
    else:
        balance = round((pos - neg) / total, 3)
        dominant = "positive" if balance > 0 else "negative" if balance < 0 else "neutral"

    return {
        "positive_count": pos,
        "negative_count": neg,
        "balance": balance,
        "dominant_sentiment": dominant,}
