import cv2
import logging
import threading
import time
import numpy as np
import os
import yaml
from flask import Flask, Response
from depth import Depth


CONFIG_PATH = "config/sgbm_params.yaml"

params = {
    "num_disparities": 336,
    "min_disparity": 48,
    "block_size": 9,
    "p1_factor": 8,
    "p2_factor": 32,
    "disp12_max_diff": 2,
    "uniqueness_ratio": 10,
    "speckle_window_size": 120,
    "speckle_range": 2,
    "pre_filter_cap": 63,
    "mode": 2,
    "proc_scale": 0.5,
    "min_depth": 0.1,
    "max_depth":0.1,
}

def load_params(config_path):
    if not os.path.exists(config_path):
        logging.warning(
            f"YAML não encontrado: {config_path}. "
            "Usando parâmetros padrão."
        )
        return params

    try:
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)

        ros_params = data["/stereo_disparity_node"]["ros__parameters"]

        for key, value in ros_params.items():
            if key in params:
                params[key] = value

        logging.info(f"Parâmetros carregados de {config_path}")

    except Exception as e:
        logging.error(f"Erro ao carregar YAML: {e}")

    return params

params = load_params(CONFIG_PATH)

scale = float(params["proc_scale"])
min_depth = float(params["min_depth"])
max_depth = float(params["max_depth"])


#Thread Lock
frame_lock = threading.Lock()
disparity_lock = threading.Lock()
image_disparity_lock = threading.Lock()

# Importa a classe principal do novo SDK
from stereo_4d import Stereo4DCameraHandler

web_feed = True
depth_calc = True
cameras_port = int(os.getenv("CAMERAS_PORT", "5000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    global latest_frames,web_feed

    while True:

        if(web_feed == False):
            time.sleep(0.033)
            continue

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
    global latest_disparity,image_latest_disparity,depth_calc
    while True:

        if(depth_calc == False):
            time.sleep(0.033)
            continue

        try:
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

            depth.process(left, right)

            with disparity_lock:
                latest_disparity = depth.get_raw_disparity().copy()

            if web_feed == True:
                with image_disparity_lock:
                    image_latest_disparity = depth.get_disparity_image().copy()


        except cv2.error as e:
            logging.warning(f"[OpenCV Error] Falha no cálculo SGBM: {e}")
            time.sleep(0.025)

        except AttributeError as e:
            logging.warning(
                f"[AttributeError] Tentativa de ler imagem/matriz nula: {e}"
            )
            time.sleep(0.05)

        except Exception as e:
            logging.error(f"[Erro Desconhecido] Na thread depth_loop: {e}")
            time.sleep(0.1)

def depth_map(min_depth_m=min_depth, max_depth_m=max_depth):

    global latest_disparity,scale

    with disparity_lock:
        if latest_disparity is None:
            return None

        disp_map = latest_disparity.copy()

    if camera.left_camera_info is None:
        return None

    # Intrínseca original da câmera
    fx_original = camera.left_camera_info.k[0, 0]

    # Ajusta fx para a resolução processada
    fx = fx_original * scale

    # Baseline
    baseline = abs(
        camera.left_camera_info.extrinsic_matrix[0, 3]
    )

    # Z = f * B / d
    depth_map = np.zeros_like(disp_map, dtype=np.float32)

    valid_mask = disp_map > 0

    depth_map[valid_mask] = (
        fx * baseline
    ) / disp_map[valid_mask]

    # Limite mínimo
    if min_depth_m is not None:
        depth_map[depth_map < min_depth_m] = 0

    # Limite máximo
    if max_depth_m is not None:
        depth_map[depth_map > max_depth_m] = 0

    return depth_map

def generate_depth_frames():
    global web_feed

    while True:

        with image_disparity_lock:
            if image_latest_disparity is not None:
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

@app.route("/video0")
def video0():
    global web_feed
    if web_feed == True:
        return Response(
            generate_frames(0),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        return "Offline"
@app.route("/video1")
def video1():
    global web_feed
    if web_feed == True:
        return Response(
            generate_frames(1),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        return "Offline"

@app.route("/disparity")
def disparity():
    global web_feed,depth_calc
    if web_feed == True and depth_calc == True:
        return Response(
            generate_depth_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        return "Offline"

@app.route("/webfeed")
def webfeed():
    global web_feed
    web_feed = not web_feed
    status = "LIGADO" if web_feed else "DESLIGADO"
    return f"Webfeed alterado para {status}", 200

@app.route("/depth-map")
def profundidade():
    return f"{depth_map()}"

app.run(
    host="0.0.0.0",
    port=cameras_port,
    threaded = True
)