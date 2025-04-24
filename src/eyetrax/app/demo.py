import time
import cv2
import numpy as np
import argparse
import os
from scipy.stats import gaussian_kde

from eyetrax.utils.screen import get_screen_size
from eyetrax.gaze import GazeEstimator
from eyetrax.calibration import (
    run_9_point_calibration,
    run_5_point_calibration,
    run_lissajous_calibration,
    fine_tune_kalman_filter,
)
from eyetrax.filters import make_kalman


def run_demo():
    parser = argparse.ArgumentParser(
        description="Gaze Estimation with Kalman Filter or KDE"
    )
    parser.add_argument("--filter", choices=["kalman", "kde", "none"], default="none")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--calibration", choices=["9p", "5p", "lissajous"], default="9p"
    )
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.5, help="0 < value < 1")
    args = parser.parse_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator()

    if calibration_method == "9p":
        run_9_point_calibration(gaze_estimator, camera_index=camera_index)
    elif calibration_method == "5p":
        run_5_point_calibration(gaze_estimator, camera_index=camera_index)
    else:
        run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    if filter_method == "kalman":
        kalman = make_kalman()
        fine_tune_kalman_filter(gaze_estimator, kalman, camera_index=camera_index)
    else:
        kalman = None

    screen_width, screen_height = get_screen_size()
    cam_width, cam_height = 320, 240
    BORDER = 2
    MARGIN = 20

    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        background[:] = (50, 50, 50)

    cv2.namedWindow("Gaze Estimation", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Gaze Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    cap = cv2.VideoCapture(camera_index)
    prev_time = time.time()

    if filter_method == "kde":
        gaze_history = []
        time_window = 0.5

    cursor_alpha = 0.0
    cursor_step = 0.05

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        features, blink_detected = gaze_estimator.extract_features(frame)
        if features is not None and not blink_detected:
            gaze_point = gaze_estimator.predict(np.array([features]))[0]
            x, y = map(int, gaze_point)

            if kalman:
                prediction = kalman.predict()
                x_pred, y_pred = map(int, prediction[:2, 0])
                x_pred = max(0, min(x_pred, screen_width - 1))
                y_pred = max(0, min(y_pred, screen_height - 1))
                measurement = np.array([[np.float32(x)], [np.float32(y)]])
                if not np.any(kalman.statePre):
                    kalman.statePre[:2] = measurement
                    kalman.statePost[:2] = measurement
                kalman.correct(measurement)
            elif filter_method == "kde":
                now = time.time()
                gaze_history.append((now, x, y))
                gaze_history = [
                    (t, gx, gy)
                    for (t, gx, gy) in gaze_history
                    if now - t <= time_window
                ]
                if len(gaze_history) > 1:
                    arr = np.array([(gx, gy) for (_, gx, gy) in gaze_history])
                    try:
                        kde = gaussian_kde(arr.T)
                        xi, yi = np.mgrid[0:screen_width:320j, 0:screen_height:200j]
                        zi = (
                            kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape).T
                        )
                        flat = zi.ravel()
                        idx = np.argsort(flat)[::-1]
                        cdf = np.cumsum(flat[idx]) / flat.sum()
                        threshold = flat[idx[np.searchsorted(cdf, confidence_level)]]
                        mask = (zi >= threshold).astype(np.uint8)
                        mask = cv2.resize(mask, (screen_width, screen_height))
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        x_pred = int(np.mean(arr[:, 0]))
                        y_pred = int(np.mean(arr[:, 1]))
                    except np.linalg.LinAlgError:
                        x_pred, y_pred = x, y
                        contours = []
                else:
                    x_pred, y_pred = x, y
                    contours = []
            else:
                x_pred, y_pred = x, y
                contours = []

            cursor_alpha = min(cursor_alpha + cursor_step, 1.0)
        else:
            x_pred = y_pred = None
            blink_detected = True
            contours = []
            cursor_alpha = max(cursor_alpha - cursor_step, 0.0)

        canvas = background.copy()

        if filter_method == "kde" and contours:
            cv2.drawContours(canvas, contours, -1, (15, 182, 242), 5)

        if x_pred is not None and y_pred is not None and cursor_alpha > 0:
            overlay = canvas.copy()
            cv2.circle(overlay, (x_pred, y_pred), 30, (0, 0, 255), -1)
            cv2.circle(overlay, (x_pred, y_pred), 25, (255, 255, 255), -1)
            cv2.addWeighted(
                overlay, cursor_alpha * 0.6, canvas, 1 - cursor_alpha * 0.6, 0, canvas
            )

        small = cv2.resize(frame, (cam_width, cam_height))
        thumb = cv2.copyMakeBorder(
            small,
            BORDER,
            BORDER,
            BORDER,
            BORDER,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        h, w = thumb.shape[:2]
        canvas[-h - MARGIN : -MARGIN, -w - MARGIN : -MARGIN] = thumb

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(
            canvas,
            f"FPS: {int(fps)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        blink_txt = "Blinking" if blink_detected else "Not Blinking"
        blink_clr = (0, 0, 255) if blink_detected else (0, 255, 0)
        cv2.putText(
            canvas,
            blink_txt,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            blink_clr,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Gaze Estimation", canvas)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
