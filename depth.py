import cv2
import numpy as np
import yaml
import logging
import os

# Tenta importar o módulo extra para o Filtro WLS (Alta Qualidade)
try:
    import cv2.ximgproc
    HAS_XIMGPROC = True
except ImportError:
    HAS_XIMGPROC = False
    logging.warning("Módulo cv2.ximgproc não encontrado. Para bordas de precisão milimétrica, instale: pip install opencv-contrib-python")

class Depth:
    def __init__(self, config_path="config/sgbm_params.yaml"):
        """
        Processador de Profundidade de Alta Precisão.
        Garante a preservação dos dados matemáticos em float32 para cálculos milimétricos.
        """
        self.disparity_color = None
        self.disparity_raw = None
        
        # Valores de fábrica extraídos diretamente do SDK (stereo_disparity_node.py)[cite: 2]
        self.params = {
            "num_disparities":    336,
            "min_disparity":       48,
            "block_size":           9,
            "p1_factor":            8,
            "p2_factor":           32,
            "disp12_max_diff":      2,
            "uniqueness_ratio":    10,
            "speckle_window_size": 120,
            "speckle_range":        2,
            "pre_filter_cap":      63,
            "mode":                 2,  # 2 equivale a cv2.STEREO_SGBM_MODE_SGBM_3WAY[cite: 2]
            "proc_scale":          0.5
        }
        
        self._load_yaml_params(config_path)
        self._rebuild_sgbm()

    def _load_yaml_params(self, config_path):
        """Lê os parâmetros de calibração do arquivo YAML no formato do ROS2[cite: 2]."""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as file:
                    data = yaml.safe_load(file)
                    
                    # Procura a estrutura específica que o ROS2 utiliza[cite: 2]
                    if data and '/stereo_disparity_node' in data:
                        node_data = data['/stereo_disparity_node']
                        if 'ros__parameters' in node_data:
                            for key, val in node_data['ros__parameters'].items():
                                if key in self.params:
                                    self.params[key] = val
                            logging.info(f"Parâmetros de alta precisão carregados de {config_path}")
            except Exception as e:
                logging.error(f"Erro ao ler {config_path}: {e}. Usando fallback do SDK.")

    def _rebuild_sgbm(self):
        """Constrói os objetos do SGBM com as regras matemáticas originais do SDK[cite: 2]."""
        bs = self.params["block_size"]
        if bs % 2 == 0:
            bs += 1
            
        nd = max(16, (self.params["num_disparities"] // 16) * 16)
        
        mode_map = {
            0: cv2.STEREO_SGBM_MODE_SGBM,
            1: cv2.STEREO_SGBM_MODE_HH,
            2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            3: cv2.STEREO_SGBM_MODE_HH4,
        }
        
        # Matcher Esquerdo (Principal)
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity      = self.params["min_disparity"],
            numDisparities    = nd,
            blockSize         = bs,
            P1                = self.params["p1_factor"] * 3 * bs ** 2, # Penaliza pequenas mudanças[cite: 2]
            P2                = self.params["p2_factor"] * 3 * bs ** 2, # Penaliza grandes mudanças[cite: 2]
            disp12MaxDiff     = self.params["disp12_max_diff"],
            uniquenessRatio   = self.params["uniqueness_ratio"],
            speckleWindowSize = self.params["speckle_window_size"],
            speckleRange      = self.params["speckle_range"],
            preFilterCap      = self.params["pre_filter_cap"],
            mode              = mode_map.get(self.params["mode"], cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        )
        
        # Configuração do Filtro WLS para suavização das bordas
        if HAS_XIMGPROC:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(1.5)

        self.min_d = self.params["min_disparity"]
        self.max_d = self.min_d + nd
        self.scale = self.params["proc_scale"]

    def process(self, left_frame, right_frame):
        """Processa os frames e gera as matrizes (Visual e Matemática)."""
        if left_frame is None or right_frame is None:
            return

        h_orig, w_orig = left_frame.shape[:2]

        # 1. Redimensionamento para performance, mantendo a regra do SDK[cite: 2]
        left_s = cv2.resize(left_frame, None, fx=self.scale, fy=self.scale)
        right_s = cv2.resize(right_frame, None, fx=self.scale, fy=self.scale)

        # 2. Conversão obrigatória para tons de cinza
        gl = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY) if len(left_s.shape) == 3 else left_s
        gr = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY) if len(right_s.shape) == 3 else right_s

        # 3. Processamento Estéreo e Filtro WLS
        if HAS_XIMGPROC:
            left_disp = self.left_matcher.compute(gl, gr)
            right_disp = self.right_matcher.compute(gr, gl)
            
            # WLS guia a profundidade usando a imagem original
            filtered_disp = self.wls_filter.filter(left_disp, left_s, None, right_disp)
            
            # Divide por 16.0 garantindo o float32 (casas decimais precisas)[cite: 2]
            disp_float = filtered_disp.astype(np.float32) / 16.0
        else:
            left_disp = self.left_matcher.compute(gl, gr)
            disp_float = left_disp.astype(np.float32) / 16.0

        # Remove valores inválidos (buracos sem textura ou bordas ocluídas)
        disp_float[disp_float < self.min_d] = 0

        # 4. SALVANDO A MATRIZ DE PRECISÃO (Sem compressão de cores)
        # Usa INTER_NEAREST para evitar a criação de distâncias falsas por interpolação
        self.disparity_raw = cv2.resize(disp_float, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 5. GERANDO A IMAGEM VISUAL (Para o Flask/Stream)
        valid = disp_float >= self.min_d
        norm = np.zeros(disp_float.shape, dtype=np.uint8)
        
        # Achata a profundidade em 255 tons[cite: 2]
        norm[valid] = np.clip(
            (disp_float[valid] - self.min_d) / (self.max_d - self.min_d) * 255.0, 
            0, 255
        ).astype(np.uint8)

        # Mapeia as cores e deixa o fundo inválido em preto (0,0,0)
        disp_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        disp_color[norm == 0] = [0, 0, 0]

        # Restaura a resolução para a tela web
        self.disparity_color = cv2.resize(disp_color, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    def get_disparity_image(self):
        """
        Retorna a imagem colorida e comprimida (8-bits). 
        Uso: Transmissão no navegador via Flask.
        """
        return self.disparity_color
        
    def get_raw_disparity(self):
        """
        Retorna a matriz de dados puros (float32). 
        Uso: Cálculos matemáticos precisos em milímetros (Z = f * B / d).
        """
        return self.disparity_raw