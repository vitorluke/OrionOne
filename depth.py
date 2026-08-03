import cv2
import numpy as np


class Depth:

    def __init__(self):

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 8,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        self.disparity = None

    def process(self, left_frame, right_frame):

        if left_frame is None or right_frame is None:
            return None

        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        self.disparity = self.stereo.compute(
            gray_left,
            gray_right
        ).astype(np.float32) / 16.0

        return self.disparity

    def get_disparity_image(self):

        if self.disparity is None:
            return None

        return cv2.normalize(
            self.disparity,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)