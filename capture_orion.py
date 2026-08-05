import cv2
import logging
import threading
import time
import numpy as np
from flask import Flask, Response
from os import getenv
from depth import Depth

#Thread Lock
frame_lock = threading.Lock()
disparity_lock = threading.Lock()
image_disparity_lock = threading.Lock()

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
image_latest_disparity = None
latest_disparity = None

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
    global latest_disparity,image_latest_disparity
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

        with disparity_lock:
            latest_disparity = depth.get_raw_disparity().copy()

        if web_feed == True:
            with image_disparity_lock:
                image_latest_disparity = depth.get_disparity_image().copy()
       
        if image_latest_disparity is not None:
            # Exibe o mapa de profundidade localmente
            cv2.imshow("Disparity", image_latest_disparity)

        if cv2.waitKey(1) == 27: # Pressione ESC para fechar a janela local
            break

def depth_map():
    """
    Calcula o mapa de profundidade completo em metros (matriz numpy float32)
    para todos os pixels, utilizando a disparidade crua e os dados do SDK da câmera.
    """
    global latest_disparity

    with disparity_lock:
        if latest_disparity is None:
            return None
        disp_map = latest_disparity.copy()

    # Verifica se as informações da câmera estão disponíveis no SDK[cite: 10]
    if camera.left_camera_info is None:
        return None

    # Extrai os parâmetros reais diretamente do objeto camera[cite: 10]
    focal_length = camera.left_camera_info.k[0, 0]  # f_x em pixels[cite: 10]
    baseline = camera.left_camera_info.extrinsic_matrix[0, 3]  # Linha de base em metros[cite: 10]

    # Cria uma matriz de zeros com o mesmo formato da disparidade
    depth_map = np.zeros_like(disp_map, dtype=np.float32)

    # Identifica pixels onde a disparidade é válida (maior que zero)
    valid_mask = disp_map > 0

    # Cálculo vetorizado para todos os pixels válidos de uma só vez: Z = (f * B) / d
    depth_map[valid_mask] = (focal_length * baseline) / disp_map[valid_mask]

    return depth_map

def generate_depth_frames():

    while True:

        with image_disparity_lock:
            if image_latest_disparity != None:
                frame = image_latest_disparity.copy()
            else:
                frame = None

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640,480))
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


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
    @app.route("/video0")
    def video0():
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

    @app.route("/disparity")
    def disparity():
        return Response(
            generate_depth_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    

    app.run(
        host="0.0.0.0",
        port=cameras_port,
        threaded=True
    )