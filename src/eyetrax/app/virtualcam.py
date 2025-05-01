import cv2
import numpy as np
import pyvirtualcam

from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames
from eyetrax.gaze import GazeEstimator
from eyetrax.calibration import (
    run_9_point_calibration,
    run_5_point_calibration,
    run_lissajous_calibration,
    fine_tune_kalman_filter,
)
from eyetrax.filters import (
    make_kalman,
    KalmanSmoother,
    KDESmoother,
    NoSmoother,
)
from eyetrax.cli import parse_common_args


def run_virtualcam():

    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator()

    if calibration_method == "9p":
        run_9_point_calibration(gaze_estimator, camera_index=camera_index)
    elif calibration_method == "5p":
        run_5_point_calibration(gaze_estimator, camera_index=camera_index)
    else:
        run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()

    if filter_method == "kalman":
        kalman = make_kalman()
        fine_tune_kalman_filter(gaze_estimator, kalman, camera_index=camera_index)
        smoother = KalmanSmoother(kalman)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    green_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    green_bg[:] = (0, 255, 0)

    with camera(camera_index) as cap:
        cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        with pyvirtualcam.Camera(
            width=screen_width,
            height=screen_height,
            fps=cam_fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera started: {cam.device}")
            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)

                if features is not None and not blink_detected:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)
                    x_pred, y_pred = smoother.step(x, y)
                    contours = smoother.debug.get("contours", [])
                else:
                    x_pred = y_pred = None
                    contours = []

                output = green_bg.copy()
                if contours:
                    cv2.drawContours(output, contours, -1, (0, 0, 255), 3)
                if x_pred is not None and y_pred is not None:
                    cv2.circle(output, (x_pred, y_pred), 10, (0, 0, 255), -1)

                cam.send(output)
                cam.sleep_until_next_frame()


if __name__ == "__main__":
    run_virtualcam()
