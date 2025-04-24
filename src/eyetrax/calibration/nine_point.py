import cv2
import numpy as np

from eyetrax.utils.screen import get_screen_size
from eyetrax.calibration.common import wait_for_face_and_countdown, _pulse_and_capture


def run_9_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    Standard nine-point calibration
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    mx, my = int(sw * 0.1), int(sh * 0.1)
    gw, gh = sw - 2 * mx, sh - 2 * my
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
    pts = [(mx + int(c * (gw / 2)), my + int(r * (gh / 2))) for (r, c) in order]

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        return
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
