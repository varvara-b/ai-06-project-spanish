import torch
import io
import streamlit as st

from transformers import pipeline
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch._C._log_api_usage_once("app")


def load_image():
    
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


st.title('Распознай испанский текст с изображения!')
img = load_image()

result = st.button('Распознать изображение')
if result:
    captioner = pipeline(
        "image-to-text",
        "microsoft/trocr-large-printed",
        token=st.secrets["HUGGINGFACE_TOKEN"],
            max_new_tokens=50
        )
    text = captioner(img)
    st.write('Результаты распознавания:')
    st.write(text[0]["generated_text"])
