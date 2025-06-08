from ultralytics import YOLO
import cv2
import time
import torch

# Verificar si GPU está disponible
print("✅ GPU disponible:", torch.cuda.is_available())

# Cargar el modelo YOLO personalizado
model = YOLO("Entrenamiento/best.pt")

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Configurar resolución y FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame de la cámara.")
        break

    # Redimensionar frame para mejorar velocidad
    frame = cv2.resize(frame, (640, 480))

    # Medir tiempo de inferencia
    start = time.time()

    # Inferencia YOLO
    results = model(frame)

    # Dibujar resultados
    annotated_frame = results[0].plot()

    # Calcular y mostrar FPS
    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar resultado
    cv2.imshow("YOLO Inference", annotated_frame)

    # Salir con ESC (tecla 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

