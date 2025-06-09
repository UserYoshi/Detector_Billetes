import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import base64
import pandas as pd

# Configurar la página
st.set_page_config(page_title="💵 Comparador de Billetes", layout="wide")

# CSS personalizado para diseño tipo comparador
st.markdown("""
<style>
.comparador-container {
    display: flex;
    justify-content: space-around;
    align-items: start;
    gap: 30px;
    margin-top: 30px;
    flex-wrap: wrap;
}

.comparador-item {
    text-align: center;
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    flex: 1;
    max-width: 45%;
}

.comparador-label {
    font-size: 20px;
    font-weight: bold;
    color: #1a4d2e;
    margin-bottom: 10px;
}

.comparador-img {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# Cargar modelo YOLO
model = YOLO("best.pt")

# Título y descripción
st.title("💵 Comparador Visual de Billetes Detectados")
st.markdown("""
Este dashboard permite subir una imagen y ver una comparativa visual entre la imagen original 
y la imagen con billetes detectados usando YOLOv8. También se muestran métricas globales y confianza.
""")

# Subida de imagen
uploaded_file = st.file_uploader("📸 Sube una imagen JPG o PNG", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Leer y convertir imagen
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Ejecutar detección
    results = model(img_array)
    annotated = results[0].plot()
    boxes = results[0].boxes

    # Convertir imágenes a base64 para mostrarlas en HTML
    # Imagen original
    buffered_original = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))[1]
    original_b64 = base64.b64encode(buffered_original).decode()

    # Imagen con detecciones
    buffered_annotated = cv2.imencode('.jpg', annotated)[1]
    annotated_b64 = base64.b64encode(buffered_annotated).decode()

    # Mostrar imágenes tipo comparador
    # Mostrar imágenes tipo comparador (horizontal)
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📷 Imagen Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("### 🧠 Imagen con Detecciones")
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

        # Conteo por clase
        for clase, cantidad in class_counts.items():
            st.write(f"🧾 **{clase}**: {cantidad} billete(s)")

        # Tabla con detalles
        df = pd.DataFrame(data)
        st.markdown("### 📋 Detalles individuales")
        st.table(df)

        # Promedio de confianza
        confs = [conf.item() for conf in boxes.conf]
        promedio = np.mean(confs)
        st.success(f"🔎 **Confianza promedio:** {promedio:.2f}")
    else:
        st.warning("⚠️ No se detectaron billetes en la imagen.")
