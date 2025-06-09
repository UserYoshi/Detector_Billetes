import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import pandas as pd

# Cargar el modelo YOLO
model = YOLO("best.pt")  # Asegúrate de tener este archivo en el mismo directorio o usar descarga con gdown

# Configurar diseño de página
st.set_page_config(page_title="💵 Detector de Billetes", layout="wide")

# Encabezado
st.title("💵 Detector de Billetes Colombianos")
st.markdown("""
Este dashboard permite subir una imagen y detectar automáticamente billetes colombianos 
utilizando una red neuronal convolucional YOLOv8 entrenada por el estudiante.  
Carga una imagen con uno o varios billetes y obtén el resultado visual junto con estadísticas útiles.
""")

# Subida de imagen
uploaded_file = st.file_uploader("📸 Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

# Si hay imagen cargada
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Realizar detección con YOLO
    results = model(img_array)
    annotated = results[0].plot()
    boxes = results[0].boxes

    # Mostrar imágenes en columnas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Imagen Original")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🧠 Imagen con Detecciones")
        st.image(annotated, use_column_width=True)

    # Métricas de detección
    st.markdown("---")
    st.subheader("📊 Estadísticas de Detección")

    if boxes is not None and boxes.cls.numel() > 0:
        class_counts = {}
        data = []

        for cls, conf in zip(boxes.cls, boxes.conf):
            cls_name = model.names[int(cls)]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            data.append({"Clase": cls_name, "Confianza": f"{conf:.2f}"})

        # Mostrar conteo por clase
        for clase, cantidad in class_counts.items():
            st.write(f"🧾 **{clase}**: {cantidad} billete(s)")

        # Mostrar tabla detallada
        df = pd.DataFrame(data)
        st.markdown("### 📋 Detalles individuales")
        st.table(df)

        # Promedio de confianza
        confs = [conf.item() for conf in boxes.conf]
        promedio = np.mean(confs)
        st.success(f"🔎 **Confianza promedio:** {promedio:.2f}")
    else:
        st.warning("⚠️ No se detectaron billetes en la imagen.")

