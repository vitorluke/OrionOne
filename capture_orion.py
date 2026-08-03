import cv2
import logging
import threading
import time
from flask import Flask, Response
from os import getenv
from depth import Depth

#Thread Lock
frame_lock = threading.Lock()

# Importa a classe principal do novo SDK
from stereo_4d import Stereo4DCameraHandler

web_feed = getenv("WEB_FEED", "True").lower() == "true"
cameras_port = int(getenv("CAMERAS_PORT", "5000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if web_feed == True:
    app = Flask(__name__)

# Armazena o último frame de cada lente (0 = Esquerda, 1 = Direita)
latest_frames = {0: None, 1: None}

# Profundidade
depth = Depth()

# ==========================================
# INICIALIZAÇÃO DA CÂMERA 4D (REDE)
# ==========================================
try:
    # Habilitamos rectify_internally=True para que o SDK alinhe as imagens automaticamente
    # usando os parâmetros intrínsecos que a câmera fornece.
    camera = Stereo4DCameraHandler(show_stream=False, rectify_internally=True)
    
    logging.info("Iniciando conexão com a câmera 4D via rede...")
    
    # Inicia a comunicação e aguarda a câmera estar pronta
    if not camera.start(wait=True, timeout=15.0):
        logging.error("Falha ao iniciar a câmera 4D. Verifique a conexão de rede.")
        exit(1)
        
    logging.info("Câmera 4D conectada com sucesso!")
except Exception as e:
    logging.error(f"Erro fatal ao instanciar a câmera: {e}")
    exit(1)


def capture_loop():
    """
    Loop único que resgata o frame combinado do SDK e o divide 
    nas lentes esquerda e direita.
    """
    global latest_frames

    while True:
        try:
            # Puxa o último frame do buffer ZMQ (retorna um objeto Stereo4DFrame)
            frame_obj = camera.get_last_frame()
            
            if frame_obj is not None and frame_obj.image is not None:
                combined_image = frame_obj.image
                
                # O SDK retorna as duas lentes concatenadas horizontalmente.
                # Dividimos a matriz da imagem na metade da largura.
                width = combined_image.shape[1]
                mid = width // 2

                with frame_lock:
                    latest_frames[0] = combined_image[:, :mid].copy()
                    latest_frames[1] = combined_image[:, mid:].copy()
        
            else:
                time.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Erro no loop de captura: {e}")
            time.sleep(0.1)


def generate_frames(camera_id):
    global latest_frames

    while True:

        with frame_lock: 
            if latest_frames[camera_id] is not None:
                frame = latest_frames[camera_id].copy()
            else:
                frame = None

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


def depth_loop():
    while True:

        with frame_lock:
            if latest_frames[0] is not None and latest_frames[1] is not None:
                left = latest_frames[0].copy()
                right = latest_frames[1].copy()
            else:
                left = None
                right = None

        if left is None or right is None:
            time.sleep(0.01)
            continue

        # Processa a profundidade usando a classe Depth inalterada
        depth.process(left, right)
        disparity = depth.get_disparity_image()

        if disparity is not None:
            # Exibe o mapa de profundidade localmente
            cv2.imshow("Disparity", disparity)

        if cv2.waitKey(1) == 27: # Pressione ESC para fechar a janela local
            break


# Cria APENAS UMA thread de captura, pois o SDK traz as duas lentes juntas
threading.Thread(
    target=capture_loop,
    daemon=True
).start()

# Mantém a thread de profundidade original
threading.Thread(
    target=depth_loop,
    daemon=True
).start()

if web_feed == True:
    @app.route("/video")
    def video():
        return Response(
            generate_frames(0),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/video1")
    def video1():
        return Response(
            generate_frames(1),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    app.run(
        host="0.0.0.0",
        port=cameras_port,
        threaded=True
    )