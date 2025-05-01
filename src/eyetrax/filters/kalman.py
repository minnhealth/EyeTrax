from __future__ import annotations

from typing import Tuple

import numpy as np

from . import make_kalman
from .base import BaseSmoother


class KalmanSmoother(BaseSmoother):

    def __init__(self, kf=None) -> None:
        super().__init__()

        try:
            import cv2

            self.kf = kf if isinstance(kf, cv2.KalmanFilter) else make_kalman()
        except ImportError:
            self.kf = make_kalman()

    def step(self, x: int, y: int) -> Tuple[int, int]:
        meas = np.array([[float(x)], [float(y)]], dtype=np.float32)

        if not np.any(self.kf.statePost):
            self.kf.statePre[:2] = meas
            self.kf.statePost[:2] = meas

        pred = self.kf.predict()
        self.kf.correct(meas)

        return int(pred[0, 0]), int(pred[1, 0])
