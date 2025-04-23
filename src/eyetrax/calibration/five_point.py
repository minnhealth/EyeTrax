import time
import cv2
import numpy as np

from eyetrax.utils.screen import get_screen_size
from eyetrax.calibration.common import wait_for_face_and_countdown


def run_5_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    Faster five-point calibration
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    mx, my = int(sw * 0.1), int(sh * 0.1)
    gw, gh = sw - 2 * mx, sh - 2 * my
    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2)]
    pts = [(mx + int(c * (gw / 2)), my + int(r * (gh / 2))) for (r, c) in order]

    feats, targs = [], []
    pulse_d, cd_d = 1.0, 1.0

    for _ in range(1):
        for x, y in pts:
            ps = time.time()
            final_radius = 20
            while True:
                e = time.time() - ps
                if e > pulse_d:
                    break
                r, f = cap.read()
                if not r:
                    continue
                c = np.zeros((sh, sw, 3), dtype=np.uint8)
                radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
                final_radius = radius
                cv2.circle(c, (x, y), radius, (0, 255, 0), -1)
                cv2.imshow("Calibration", c)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cs = time.time()
            while True:
                e = time.time() - cs
                if e > cd_d:
                    break
                r, f = cap.read()
                if not r:
                    continue
                c = np.zeros((sh, sw, 3), dtype=np.uint8)
                cv2.circle(c, (x, y), final_radius, (0, 255, 0), -1)
                t = e / cd_d
                ease = t * t * (3 - 2 * t)
                ang = 360 * (1 - ease)
                cv2.ellipse(c, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
                cv2.imshow("Calibration", c)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                ft, blink = gaze_estimator.extract_features(f)
                if ft is not None and not blink:
                    feats.append(ft)
                    targs.append([x, y])

    cap.release()
    cv2.destroyAllWindows()
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
