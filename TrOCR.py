import torch
import io
import streamlit as st
from transformers import pipeline
from PIL import Image

def load_image():
    """Загрузка изображения через интерфейс Streamlit"""
    uploaded_file = st.file_uploader(
        label='Загрузите изображение с испанским текстом',
        type=['jpg', 'jpeg', 'png'],
        help="Изображение должно содержать четкий текст (рекомендуемое разрешение 300dpi)"
    )
    if uploaded_file is not None:
        try:
            image_data = uploaded_file.getvalue()
            st.image(image_data, caption='Ваше изображение', use_column_width=True)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            st.error(f"Ошибка загрузки изображения: {str(e)}")
            return None
    return None


# Конфигурация страницы
st.set_page_config(page_title="TrOCR Spanish Text Recognition", layout="centered")
st.title('TrOCR: Распознавание испанского текста')

# Информация о модели
with st.expander("О модели TrOCR"):
    st.markdown(""" 
    **TrOCR** (Transformer-based Optical Character Recognition) от Microsoft:
    - Специализированная модель для OCR
    - Поддержка печатного и рукописного текста
    - Оптимизирована для испанского языка
    - Основана на архитектуре Transformer
    """)

# Загрузка изображения
img = load_image()

if st.button('Распознать текст', disabled=img is None):
    with st.spinner('TrOCR обрабатывает ваш текст...'):
        try:
            # Инициализация пайплайна с TrOCR для испанского
            captioner = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-printed",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Обработка изображения
            text = captioner(img)

            # Вывод результатов
            st.subheader("Результат распознавания:")
            st.success(text[0]["generated_text"])

            # Дополнительная информация
            with st.expander("Технические детали"):
                st.json({
                    "Модель": "microsoft/trocr-base-printed",
                    "Версия": "base",
                    "Тип текста": "печатный",
                    "Язык": "испанский",
                    "Устройство": "CUDA" if torch.cuda.is_available() else "CPU"
                })

        except Exception as e:
            st.error(f"Ошибка распознавания: {str(e)}")
            st.info(""" 
            Советы для лучшего распознавания:
            1. Убедитесь, что текст четкий и хорошо освещен
            2. Попробуйте выровнять изображение
            3. Для рукописного текста используйте модель microsoft/trocr-base-handwritten
            """)

if img is None:
    st.warning("Пожалуйста, загрузите изображение с текстом")

# Рекомендации по выбору модели
st.markdown("### Доступные версии TrOCR:")
st.table({
    "Модель": ["microsoft/trocr-base-printed", "microsoft/trocr-base-handwritten", "microsoft/trocr-large-printed"],
    "Тип текста": ["Печатный", "Рукописный", "Печатный (большая модель)"],
    "Языки": ["Многоязычная", "Многоязычная", "Многоязычная"]
})
