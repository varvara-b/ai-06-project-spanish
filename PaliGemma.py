# Для работы этого кода потребуется: pip install torch transformers streamlit

import io
import streamlit as st
from transformers import pipeline
from PIL import Image

def load_image():
    """Загрузка изображения через интерфейс Streamlit"""
    uploaded_file = st.file_uploader(label='Загрузите изображение с испанским текстом', 
                                   type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption='Загруженное изображение', use_column_width=True)
        return Image.open(io.BytesIO(image_data))
    return None

# Конфигурация страницы
st.set_page_config(page_title="PaliGemma Spanish Text Recognition", layout="wide")
st.title('PaliGemma: Распознавание испанского текста с изображений')

# Информация о модели
with st.expander("О модели PaliGemma"):
    st.markdown("""
    **PaliGemma** — современная мультимодальная модель от Google, способная:
    - Анализировать изображения
    - Распознавать текст на нескольких языках
    - Отвечать на вопросы об изображении
    - Выполнять OCR с высокой точностью
    
    *Модель поддерживает испанский язык среди других основных языков.*
    """)

# Загрузка изображения
img = load_image()

if img and st.button('Распознать текст', type='primary'):
    with st.spinner('Модель PaliGemma обрабатывает изображение...'):
        try:
            # Инициализация пайплайна с PaliGemma
            pipe = pipeline(
                "image-to-text",
                model="google/paligemma-3b-mix-224",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Обработка изображения (добавляем испанский промпт для лучшего результата)
            prompt = "¿Qué texto aparece en esta imagen? Responde solo con el texto reconocido."
            inputs = pipe.preprocess(img, prompt=prompt, return_tensors="pt")
            
            # Генерация текста
            outputs = pipe.generate(**inputs)
            text = pipe.postprocess(outputs)[0]["generated_text"]
            
            # Вывод результатов
            st.subheader("Результат распознавания:")
            st.success(text)
            
            # Дополнительная информация
            with st.expander("Технические детали"):
                st.json({
                    "Модель": "google/paligemma-3b-mix-224",
                    "Разрешение": "224x224",
                    "Язык": "Испанский",
                    "Устройство": "CUDA" if torch.cuda.is_available() else "CPU"
                })
                
        except Exception as e:
            st.error(f"Ошибка при обработке: {str(e)}")
            st.info("Попробуйте другое изображение или проверьте его качество.")
elif not img:
    st.warning("Пожалуйста, загрузите изображение перед распознаванием")

# Добавляем примеры для тестирования
st.markdown("### Примеры для тестирования:")
col1, col2 = st.columns(2)
with col1:
    st.image("https://i.imgur.com/JQ7w7gk.jpg", caption="Пример 1: Печатный текст")
with col2:
    st.image("https://i.imgur.com/5V5QW9x.jpg", caption="Пример 2: Рукописные заметки")