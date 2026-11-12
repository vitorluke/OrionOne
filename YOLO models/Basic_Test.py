
import cv2
from ultralytics import YOLO

# Carregar modelo YOLO
model = YOLO("hole.pt")


def capture(ret, frame):
    # Executar detecção
    results = model(frame)

    # Desenhar caixas e identificações
    annotated_frame = results[0].plot()

    # Exibir imagem
    cv2.imshow("YOLO + OpenCV", annotated_frame)



