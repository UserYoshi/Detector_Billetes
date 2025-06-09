import streamlit as st
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

# Cargar el modelo YOLO
model = YOLO("best.pt")  # asegúrate que el archivo esté en el mismo repo o usa un enlace externo

st.set_page_config(page_title="Detector de Billetes", layout="wide")
st.title("🧠 Detector de Billetes")

# Subida de imagen
uploaded_file = st.file_uploader("📸 Sube una imagen para analizar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir a imagen OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Mostrar la imagen original
    st.image(image, caption="Imagen original", use_column_width=True)

    # Procesar con YOLO
    results = model(img_array)

    # Dibujar los resultados
    annotated_frame = results[0].plot()

    # Mostrar imagen con detecciones
    st.image(annotated_frame, caption="🔍 Detecciones", use_column_width=True)

    # Mostrar métricas
    st.subheader("📊 Métricas de detección")
    boxes = results[0].boxes
    if boxes is not None:
        class_counts = {}
        for cls in boxes.cls:
            cls_name = model.names[int(cls)]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        for cls_name, count in class_counts.items():
            st.write(f"- {cls_name}: {count}")
    else:
        st.write("No se detectaron objetos.")
