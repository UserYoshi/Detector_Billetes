import streamlit as st

from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("💵 Detector de Billetes Colombianos")

# Cargar modelo YOLO
model = YOLO("modelo.pt")  # Asegúrate de subir el .pt al repositorio

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir a array
    img_array = np.array(image)

    # Ejecutar predicción
    results = model.predict(img_array)

    # Mostrar resultado con cajas
    for r in results:
        im_array = r.plot()  # dibuja los bounding boxes
        st.image(im_array, caption="Predicción YOLO", use_column_width=True)

