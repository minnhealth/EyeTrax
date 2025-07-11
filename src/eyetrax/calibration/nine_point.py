import cv2
import numpy as np

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


def run_9_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    Standard nine-point calibration
    """
    # --- enforce correct class ---
    if not isinstance(gaze_estimator, GazeEstimator):
        raise TypeError(
            f"run_9_point_calibration() requires a "
            f"GazeEstimator, but got {type(gaze_estimator).__name__}"
        )

    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    order = [
        (1, 1),
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
    ]
    pts = compute_grid_points(order, sw, sh)

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        return
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))


# --- Multi-eye 9-point ---
def run_9_point_calibration_multi_eye(gaze_estimator, camera_index: int = 0):
    """
    Standard nine-point calibration (multi-eye). Returns whether or not the calibration completed.
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
        cap.release(); cv2.destroyAllWindows(); return False
    order = [(1,1),(0,0),(2,0),(0,2),(2,2),(1,0),(0,1),(2,1),(1,2)]
    pts = compute_grid_points(order, sw, sh)
    res = _pulse_and_capture_multi_eye(cap, pts, sw, sh, gaze_estimator)
    cap.release(); cv2.destroyAllWindows()
    if res is None:
        return False
    feats_l, feats_r, targs = res
    if feats_l:
        gaze_estimator.train(
            np.vstack(feats_l), np.vstack(targs),
            np.vstack(feats_r), np.vstack(targs)
        )
    return True