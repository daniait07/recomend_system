import json
from typing import Set, Tuple, List
from providers.models import Entity


def parse_ground_truth(raw: str) -> List[Entity]:
    """
    Поддерживает формат: {"entities":[{"text":"...","label":"PERSON"}, ...]}
    """
    raw_strip = raw.strip()

    data = json.loads(raw_strip)
    ents = []
    if (
        isinstance(data, dict)
        and "entities" in data
        and isinstance(data["entities"], list)
    ):
        for entity in data["entities"]:
            if isinstance(entity, dict) and "text" in entity and "label" in entity:
                ents.append(
                    Entity(text=str(entity["text"]), label=str(entity["label"]))
                )
        return ents

    return ents

def normalize_entity(entity: Entity) -> Tuple[str, str]:
    return (entity.text.strip().lower(), entity.label.strip().upper())

def evaluate_sets(gt: Set[Tuple[str, str]], pred: Set[Tuple[str, str]]):
    matched = gt & pred
    missed = gt - pred
    spurious = pred - gt

    tp = len(matched)
    fp = len(spurious)
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    scores = {"precision": precision, "recall": recall, "f1": f1}
    return scores, matched, missed, spurious
