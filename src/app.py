import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
from PIL import Image
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.model_builder import create_turtle_model

WEIGHTS_PATH = os.path.join(project_root, "turtle_weights.weights.h5")
PKL_PATH = os.path.join(project_root, "class_names.pkl")

st.set_page_config(page_title="Deniz Kaplumbağası Tanıma", page_icon="🐢", layout="centered")

@st.cache_resource
def load_turtle_model():
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(PKL_PATH):
        return None, "Ağırlıklar (.h5) veya Sınıf İsimleri (.pkl) bulunamadı. Lütfen önce eğitimi tamamlayın."
    
    try:
        with open(PKL_PATH, "rb") as f:
            class_names = pickle.load(f)
        
        model, _ = create_turtle_model(num_classes=len(class_names))
        
        model.load_weights(WEIGHTS_PATH)
        
        return (model, class_names), None
    except Exception as e:
        return None, f"Model yükleme hatası: {str(e)}"

st.title("🐢 Deniz Kaplumbağası Tanıma Sistemi")
st.markdown("---")

model_data, err = load_turtle_model()

if err:
    st.error(err)
    st.info("💡 Not: Eğer 'deserialization' hatası alıyorsanız, train_ai.py dosyasını güncelleyip 'model.save_weights()' ile ağırlıkları tekrar kaydetmelisiniz.")
    st.stop()
else:
    model, class_names = model_data

uploaded_file = st.file_uploader("Bir kaplumbağa fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
    
    if st.button('Analiz Et'):
        with st.spinner('Yapay zeka türü teşhis ediyor...'):
            try:
                img = image.convert('RGB').resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                result_index = np.argmax(predictions[0])
                confidence = predictions[0][result_index]
                predicted_class = class_names[result_index]

                st.subheader("Analiz Sonucu")
                if confidence >= 0.50:
                    st.success(f"Bu bir **{predicted_class}**")
                    st.metric("Güven Oranı", f"%{confidence * 100:.2f}")
                else:
                    st.warning(f"Tür Tahmini: {predicted_class}")
                    st.info(f"Güven Oranı (%{confidence * 100:.2f}) düşük. Daha net bir görsel deneyin.")
                
                chart_data = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
                st.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Tahmin hatası: {e}")
else:
    st.info("Lütfen bir görsel seçerek 'Analiz Et' butonuna tıklayın.")

st.markdown("---")
st.caption("Bartu Sarı - Pamukkale Üniversitesi Bilgisayar Mühendisliği")