import cv2
import numpy as np


class Depth:

    def __init__(self):

        # Parâmetros da calibração
        self.fx = 2284.4439266807667
        self.baseline = 0.29331903822053373

        # CLAHE para melhorar contraste
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        # Matcher esquerdo
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 12,
            blockSize=7,

            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,

            disp12MaxDiff=1,
            uniquenessRatio=15,

            speckleWindowSize=200,
            speckleRange=2,

            preFilterCap=63,

            mode=cv2.STEREO_SGBM_MODE_HH
        )

        # Matcher direito
        self.right_matcher = cv2.ximgproc.createRightMatcher(
            self.left_matcher
        )

        # Filtro WLS
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            self.left_matcher
        )

        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

        self.disparity = None
        self.depth = None

    def process(self, left_frame, right_frame):

        if left_frame is None or right_frame is None:
            return None

        gray_left = cv2.cvtColor(
            left_frame,
            cv2.COLOR_BGR2GRAY
        )

        gray_right = cv2.cvtColor(
            right_frame,
            cv2.COLOR_BGR2GRAY
        )

        gray_left = self.clahe.apply(gray_left)
        gray_right = self.clahe.apply(gray_right)

        left_disp = self.left_matcher.compute(
            gray_left,
            gray_right
        )

        right_disp = self.right_matcher.compute(
            gray_right,
            gray_left
        )

        disparity = self.wls_filter.filter(
            left_disp,
            gray_left,
            disparity_map_right=right_disp
        )

        disparity = disparity.astype(np.float32) / 16.0

        disparity = cv2.medianBlur(disparity, 5)

        disparity[disparity <= 0] = np.nan

        self.disparity = disparity

        depth = np.full(
            disparity.shape,
            np.nan,
            dtype=np.float32
        )

        valid = np.isfinite(disparity)

        depth[valid] = (
            self.fx * self.baseline
        ) / disparity[valid]

        self.depth = depth

        return depth

    def get_disparity(self):
        return self.disparity

    def get_depth(self):
        return self.depth

    def get_disparity_image(self):

        if self.disparity is None:
            return None

        disp = np.nan_to_num(self.disparity)

        disp = cv2.normalize(
            disp,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return cv2.applyColorMap(
            disp,
            cv2.COLORMAP_TURBO
        )

    def get_depth_image(self, max_depth=10.0):

        if self.depth is None:
            return None

        depth = self.depth.copy()

        depth[np.isnan(depth)] = max_depth
        depth = np.clip(depth, 0, max_depth)

        depth = max_depth - depth

        depth = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return cv2.applyColorMap(
            depth,
            cv2.COLORMAP_TURBO
        )

    def get_point_distance(self, x, y):

        if self.depth is None:
            return None

        if (
            x < 0 or
            y < 0 or
            x >= self.depth.shape[1] or
            y >= self.depth.shape[0]
        ):
            return None

        d = self.depth[y, x]

        if np.isnan(d):
            return None

        return float(d)