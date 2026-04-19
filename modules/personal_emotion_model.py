from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = Path("models") / "personal_emotion_model.npz"
IMAGE_SIZE = 48


class PersonalEmotionModel:
    def __init__(self, labels, centroids):
        self.labels = labels
        self.centroids = centroids

    @classmethod
    def load(cls, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=False)
        return cls(data["labels"], data["centroids"])

    def predict(self, face_gray):
        features = extract_features(face_gray)
        distances = np.linalg.norm(self.centroids - features, axis=1)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        confidence = distance_to_confidence(best_distance)

        if confidence < 45:
            return "neutral", f"personal unsure {confidence:.0f}"

        return str(self.labels[best_index]), f"personal {confidence:.0f}"


def extract_features(face_gray):
    face_gray = cv2.equalizeHist(face_gray)
    face_gray = cv2.resize(face_gray, (IMAGE_SIZE, IMAGE_SIZE))
    features = face_gray.astype(np.float32).flatten() / 255.0
    norm = np.linalg.norm(features)

    if norm > 0:
        features = features / norm

    return features


def distance_to_confidence(distance):
    confidence = 100 - (distance * 100)
    return max(0, min(100, confidence))
