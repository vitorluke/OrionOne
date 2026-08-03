import cv2
import numpy as np


class Depth:

    def __init__(self):

        # Parâmetros da calibração
        self.fx = 2284.4439266807667
        self.baseline = 0.29331903822053373

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,

            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,

            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,

            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.disparity = None
        self.depth = None

    def process(self, left_frame, right_frame):

        if left_frame is None or right_frame is None:
            return None

        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(
            gray_left,
            gray_right
        ).astype(np.float32) / 16.0

        disparity[disparity <= 0] = np.nan

        self.disparity = disparity

        depth = np.full(disparity.shape, np.nan, dtype=np.float32)

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

        return cv2.normalize(
            disp,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    def get_depth_image(self, max_depth=10.0):

        if self.depth is None:
            return None

        depth = self.depth.copy()

        depth[np.isnan(depth)] = max_depth
        depth = np.clip(depth, 0, max_depth)

        # Objetos próximos ficam mais claros
        depth = max_depth - depth

        return cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

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

        distance = self.depth[y, x]

        if np.isnan(distance):
            return None

        return float(distance)