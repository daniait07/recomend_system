import json
import streamlit as st

from settings import settings
from providers.google_gemini_ner import GoogleGeminiNER
from providers.nlpcloud_ner import NLP_CLOUD_NER
from utils.metrics import (
    parse_ground_truth,
    normalize_entity,
    evaluate_sets,
)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENTITIES_N = 5

NLP_CLOUD_MODEL = "en_core_web_lg" 
NLP_CLOUD_GPU = False
NLP_CLOUD_LANG_PREFIX = None
NLP_CLOUD_TIMEOUT = 15.0

PAGE_TITLE = "NER API Comparator"
BUTTON_TEXT = "Запустить сравнение"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

st.markdown("## 1) Входные данные")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Текст для анализа")
    input_text = st.text_area(
        "Вставьте текст или загрузите файл ниже",
        height=180,
    )

with col2:
    st.markdown("#### Эталон (ground truth) опционально, JSON-объект")
    gt_json_text = st.text_area(
        "Вставьте JSON-объект с ключом entities (или оставьте пустым)",
        height=180,
        key="gt_json_text",
    )

st.markdown("## 2) Запуск")
run_btn = st.button(BUTTON_TEXT, type="primary", use_container_width=True)

if run_btn:
    if not input_text.strip():
        st.error("Введите или загрузите текст для анализа.")
        st.stop()

    gt_entities = None
    if gt_json_text.strip():
        try:
            gt_obj = json.loads(gt_json_text)
            if not isinstance(gt_obj, dict) or "entities" not in gt_obj or not isinstance(gt_obj["entities"], list):
                raise ValueError("Ожидается словарь с ключом 'entities' и списком значений.")
            gt_entities = parse_ground_truth(json.dumps(gt_obj, ensure_ascii=False))
            st.success(f"Эталон загружен: {len(gt_entities)} сущ.")
        except Exception as e:
            st.error(f"Не удалось разобрать ground truth: {e}")
            st.info("Продолжаем без метрик, так как GT невалиден.")
            gt_entities = None
    else:
        st.info("GT не задан. Будут показаны только извлечённые сущности (без метрик).")

    providers = []

    try:
        providers.append(
            GoogleGeminiNER(
                api_key=settings.GEMINI_API_KEY,
                model=GEMINI_MODEL,
            )
        )
    except Exception as e:
        st.error(f"Ошибка инициализации Gemini: {e}")

    try:
        providers.append(
            NLP_CLOUD_NER(
                url=settings.NLP_CLOUD_URL,
                api_key=settings.NLP_CLOUD_API_KEY,
                model=NLP_CLOUD_MODEL,
                timeout=NLP_CLOUD_TIMEOUT,
            )
        )
    except Exception as e:
        st.error(f"Ошибка инициализации NLP Cloud: {e}")

    if not providers:
        st.stop()

    st.markdown("## 3) Результаты")
    gt_set = {normalize_entity(entity) for entity in gt_entities} if gt_entities else None

    for prov in providers:
        with st.spinner(f"Запрос к {prov.get_name()}..."):
            try:
                entities = prov.extract(input_text)
            except Exception as e:
                st.error(f"{prov.get_name()}: ошибка вызова {e}")
                continue

        st.subheader(f"Провайдер: {prov.get_name()}")
        st.caption(f"Извлечено сущностей: {len(entities)}")

        st.dataframe(
            [{"text": entity.text, "label": entity.label} for entity in entities],
            use_container_width=True,
        )

        if gt_set is not None:
            pred_set = {normalize_entity(entity) for entity in entities}
            scores, matched, missed, spurious = evaluate_sets(gt_set, pred_set)

            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{scores['precision']:.3f}")
            c2.metric("Recall", f"{scores['recall']:.3f}")
            c3.metric("F1", f"{scores['f1']:.3f}")

            with st.expander("Совпадения"):
                st.write(sorted(list(matched)))
            with st.expander("Пропуски (в GT, но не найдены)"):
                st.write(sorted(list(missed)))
            with st.expander("Лишние (в предсказании, но не в GT)"):
                st.write(sorted(list(spurious)))
        else:
            st.info("GT отсутствует - метрики не рассчитываются.")
