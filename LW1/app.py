import json
import streamlit as st
from settings import settings
from additional.google_gemini_ner import GoogleGeminiNER
from additional.nlpcloud_ner import NLP_CLOUD_NER
from metric.metrics import parse_ground_truth, calculate_emotion_metrics
from additional.models import Entity

PAGE_TITLE = "Эмоциональный анализ отзывов"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.title(PAGE_TITLE)
st.markdown(
    "Приложение извлекает ключевые объекты и эмоции из текста, классифицируя их по категориям:\n"
    "- TARGET: объект, о котором идет речь\n"
    "- POSITIVE_FEATURE: положительная характеристика\n"
    "- NEGATIVE_FEATURE: отрицательная характеристика\n"
    "- USER_EXPERIENCE: действия или эмоции пользователя")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Текст для анализа")
    input_text = st.text_area("Вставьте текст отзыва или комментария", height=200)

with col2:
    st.subheader("Эталонные метки (JSON)")
    gt_json_text = st.text_area(
        "Например: {\"entities\":[{\"text\":\"Toyota\",\"label\":\"TARGET\"}, ...]}",
        height=200
    )

run_btn = st.button("Анализировать эмоции", type="primary")

if run_btn:
    if not input_text.strip():
        st.error("Введите текст для анализа.")
        st.stop()

    gt_entities: list[Entity] | None = None
    if gt_json_text.strip():
        try:
            gt_obj = json.loads(gt_json_text)
            gt_entities = parse_ground_truth(json.dumps(gt_obj, ensure_ascii=False))
            st.success(f"Эталон загружен: {len(gt_entities)} сущностей")
        except Exception as e:
            st.error(f"Не удалось разобрать эталон: {e}")
            gt_entities = None
    providers = []
    try:
        providers.append(
            GoogleGeminiNER(
                api_key=settings.GEMINI_API_KEY
            )
        )
    except Exception as e:
        st.error(f"Ошибка инициализации Gemini: {e}")

    try:
        providers.append(
            NLP_CLOUD_NER(
                url=settings.NLP_CLOUD_URL,
                api_key=settings.NLP_CLOUD_API_KEY
            )
        )
    except Exception as e:
        st.error(f"Ошибка инициализации NLP Cloud: {e}")

    if not providers:
        st.stop()
    for prov in providers:
        with st.expander(f"Результаты модели: {prov.get_name()}", expanded=True):
            with st.spinner(f"Обрабатываем текст..."):
                try:
                    entities = prov.extract(input_text)
                except Exception as e:
                    st.error(f"{prov.get_name()}: ошибка вызова {e}")
                    continue

            if not entities:
                st.info("Сущности не найдены.")
                continue

            st.subheader("Извлеченные сущности")
            st.dataframe(
                [{"Текст": e.text, "Категория": e.label} for e in entities],
                use_container_width=True
            )

            metrics = calculate_emotion_metrics(entities)
            st.subheader("Эмоциональные метрики")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Позитивные характеристики", metrics["positive_count"])
            c2.metric("Негативные характеристики", metrics["negative_count"])
            c3.metric("Баланс эмоций", f"{metrics['balance']:+.2f}")
            c4.metric("Преобладающее настроение", metrics["dominant_sentiment"].capitalize())

            if gt_entities:
                st.subheader("Сравнение с эталоном")
                gt_set = {(e.text.lower(), e.label) for e in gt_entities}
                pred_set = {(e.text.lower(), e.label) for e in entities}

                matched = gt_set & pred_set
                missed = gt_set - pred_set
                spurious = pred_set - gt_set

                st.markdown(f"**Совпавшие сущности:** {len(matched)}")
                st.write(sorted(list(matched)))

                st.markdown(f"**Пропущенные (в эталоне, но не найдены):** {len(missed)}")
                st.write(sorted(list(missed)))

                st.markdown(f"**Лишние (найдены, но нет в эталоне):** {len(spurious)}")
                st.write(sorted(list(spurious)))

