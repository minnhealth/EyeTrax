from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
from eyetrax.models import BaseModel, create_model


class GazeEstimator:
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float = 0.8,
        min_history: int = 15,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.model: BaseModel = create_model(model_name, **(model_kwargs or {}))

        self._ear_history = deque(maxlen=ear_history_len)
        self._blink_ratio = blink_threshold_ratio
        self._min_history = min_history

    def extract_features(self, image):
        """
        Takes in image and returns landmarks around the eye region
        Normalization with nose tip as anchor
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, False

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )
        nose_anchor = all_points[4]
        left_corner = all_points[33]
        right_corner = all_points[263]
        top_of_head = all_points[10]

        shifted_points = all_points - nose_anchor
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = top_of_head - nose_anchor
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        left_corner_rot = R.T @ (left_corner - nose_anchor)
        right_corner_rot = R.T @ (right_corner - nose_anchor)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Blink detection
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2
        blink_detected = EAR < thr

        return features, blink_detected

    def save_model(self, path: str | Path):
        """
        Pickle model
        """
        self.model.save(path)

    def load_model(self, path: str | Path):
        self.model = BaseModel.load(path)

    def train(self, X, y, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.model.train(X, y, variable_scaling)

    def predict(self, X):
        """
        Predicts gaze location
        """
        return self.model.predict(X)

class GazeEstimatorMultiEye:
    """
    Two-eye gaze estimator: independent models per eye.
    """
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float = 0.8,
        min_history: int = 15,
    ):
        # Face mesh detector
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        # Two regressors: left and right
        self.model_left: BaseModel = create_model(model_name, **(model_kwargs or {}))
        self.model_right: BaseModel = create_model(model_name, **(model_kwargs or {}))

        # Blink/EAR history
        self._ear_history = deque(maxlen=ear_history_len)
        self._blink_ratio = blink_threshold_ratio
        self._min_history = min_history

    def _eye_features(self, rotated_points, euler, side: str):
        """
        Extracts a flat feature vector for one eye ("left" or "right").
        """
        idx = LEFT_EYE_INDICES + MUTUAL_INDICES if side=="left" else RIGHT_EYE_INDICES + MUTUAL_INDICES
        lm = rotated_points[idx]
        feats = lm.flatten()
        feats = np.concatenate([feats, euler])
        return feats

    def _compute_common(self, image) -> tuple[np.ndarray, list[float], list[mp.framework.formats.landmark_pb2.NormalizedLandmark]]:
        """
        Runs MediaPipe, normalizes landmarks, computes R, rotated_points, yaw/pitch/roll.
        Used by multi-eye extractor.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, False, None
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        all_points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
        nose_anchor = all_points[4]
        left_corner = all_points[33]
        right_corner = all_points[263]
        top_of_head = all_points[10]

        shifted_points = all_points - nose_anchor
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = top_of_head - nose_anchor
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T
        left_corner_rot = R.T @ (left_corner - nose_anchor)
        right_corner_rot = R.T @ (right_corner - nose_anchor)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll  = np.arctan2(R[2, 1], R[2, 2])
        euler = [yaw, pitch, roll]
        return rotated_points, euler, landmarks

    def extract_features(self, image) -> tuple[np.ndarray | None, np.ndarray | None, bool, bool]:
        """
        Returns: (feats_left, feats_right, blink_left, blink_right).
        """
        rotated_points, euler, landmarks = self._compute_common(image)
        if rotated_points is None:
            return None, None, True, True
        # Blink detection per eye
        left_eye_inner   = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer   = np.array([landmarks[33].x,  landmarks[33].y])
        left_eye_top     = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom  = np.array([landmarks[145].x, landmarks[145].y])
        right_eye_inner  = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer  = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top    = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])
        left_width  = np.linalg.norm(left_eye_outer  - left_eye_inner)
        left_height = np.linalg.norm(left_eye_top    - left_eye_bottom)
        right_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_height= np.linalg.norm(right_eye_top   - right_eye_bottom)
        left_EAR  = left_height  / (left_width  + 1e-9)
        right_EAR = right_height / (right_width + 1e-9)
        EAR = (left_EAR + right_EAR)/2
        self._ear_history.append(EAR)
        thr = float(np.mean(self._ear_history))*self._blink_ratio if len(self._ear_history)>=self._min_history else 0.2
        blink_left = blink_right = (EAR < thr)
        feats_left  = self._eye_features(rotated_points, euler, "left")
        feats_right = self._eye_features(rotated_points, euler, "right")
        return feats_left, feats_right, blink_left, blink_right

    def train(self, X_left: np.ndarray, y_left: np.ndarray, X_right: np.ndarray, y_right: np.ndarray) -> None:
        """Train left and right models separately."""
        self.model_left.train(X_left, y_left)
        self.model_right.train(X_right, y_right)

    def predict_left(self, X: np.ndarray) -> np.ndarray:
        """Predict left-eye gaze points."""
        return self.model_left.predict(X)

    def predict_right(self, X: np.ndarray) -> np.ndarray:
        """Predict right-eye gaze points."""
        return self.model_right.predict(X)

    def save_model(self, left_path: str | Path, right_path: str | Path) -> None:
        """Persist both regressors."""
        self.model_left.save(str(left_path))
        self.model_right.save(str(right_path))

    def load_model(self, left_path: str | Path, right_path: str | Path) -> None:
        """Load both regressors."""
        self.model_left = BaseModel.load(str(left_path))
        self.model_right = BaseModel.load(str(right_path))