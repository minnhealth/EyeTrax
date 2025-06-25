import cv2
import numpy as np
import time

from eyetrax.calibration.common import (
    compute_grid_points,
    wait_for_face_and_countdown,
    _pulse_and_capture,
    wait_for_face_and_countdown_multi_eye,
    _pulse_and_capture_multi_eye,
)
from eyetrax.utils.screen import get_screen_size
from eyetrax.gaze import GazeEstimatorMultiEye
from eyetrax.gaze import GazeEstimator


def run_16_point_calibration_multi_eye(gaze_estimator, camera_index: int = 0):
    """
    Standard sixteen-point calibration (multi-eye), on a 4×4 grid.
    """
    # --- enforce correct class ---
    if not isinstance(gaze_estimator, GazeEstimatorMultiEye):
        raise TypeError(
            f"run_9_point_calibration_multi_eye() requires a "
            f"GazeEstimatorMultiEye, but got {type(gaze_estimator).__name__}"
        )

    sw, sh = get_screen_size()
    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown_multi_eye(cap, gaze_estimator, sw, sh, 2):
        cap.release(); cv2.destroyAllWindows()
        return

    # build a 4×4 grid of normalized (row, col) indices
    rows, cols = 4, 5
    order = [(r, c) for r in range(rows) for c in range(cols)]
    # optional: sort center-out for smoother UX
    center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
    order.sort(key=lambda rc: abs(rc[0] - center_r) + abs(rc[1] - center_c))

    # map to pixel coords
    pts = compute_grid_points(order, sw, sh)

    res = _pulse_and_capture_multi_eye(cap, pts, sw, sh, gaze_estimator)
    cap.release(); cv2.destroyAllWindows()
    if res is None:
        return

    feats_l, feats_r, targs = res
    if feats_l:
        gaze_estimator.train(
            np.vstack(feats_l), np.vstack(targs),
            np.vstack(feats_r), np.vstack(targs)
        )