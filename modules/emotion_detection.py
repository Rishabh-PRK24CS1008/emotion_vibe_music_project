
from collections import Counter, deque

import cv2

from modules.personal_emotion_model import PersonalEmotionModel

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

DEEPFACE_ERROR = None

try:
    from deepface import DeepFace
except Exception as exc:
    DeepFace = None
    DEEPFACE_ERROR = exc

EMOTION_ALIASES = {
    "fear": "sad",
    "surprise": "neutral",
    "disgust": "angry",
}
MODEL_INTERVAL_FRAMES = 8
MIN_CONFIDENCE = {
    "happy": 80,
    "sad": 35,
    "angry": 35,
    "neutral": 45,
}
MIN_MARGIN = 12
frame_count = 0
last_model_emotion = None
personal_model = PersonalEmotionModel.load()


class EmotionSmoother:
    def __init__(self, window_size=8, min_votes=4):
        self.window = deque(maxlen=window_size)
        self.current = "neutral"
        self.min_votes = min_votes

    def update(self, emotion):
        self.window.append(emotion)
        emotion, votes = Counter(self.window).most_common(1)[0]

        if votes >= self.min_votes:
            self.current = emotion

        return self.current


smoother = EmotionSmoother()


def normalize_emotion(emotion):
    emotion = (emotion or "neutral").lower()
    emotion = EMOTION_ALIASES.get(emotion, emotion)

    if emotion not in {"happy", "sad", "angry", "neutral"}:
        return "neutral"

    return emotion


def normalize_scores(scores):
    normalized = Counter()

    for emotion, score in scores.items():
        normalized[normalize_emotion(emotion)] += float(score)

    return dict(normalized)


def choose_model_emotion(scores):
    scores = normalize_scores(scores)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if not ranked:
        return "neutral", "no scores"

    emotion, confidence = ranked[0]
    second_confidence = ranked[1][1] if len(ranked) > 1 else 0
    margin = confidence - second_confidence

    if confidence < MIN_CONFIDENCE.get(emotion, 45):
        return "neutral", f"{emotion} weak {confidence:.0f}"

    if margin < MIN_MARGIN and emotion != "neutral":
        return "neutral", f"{emotion} close {confidence:.0f}/{second_confidence:.0f}"

    return emotion, f"{emotion} {confidence:.0f}"


def detect_with_deepface(face):
    if DeepFace is None:
        return None

    try:
        result = DeepFace.analyze(
            face,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return None

    if isinstance(result, list):
        result = result[0]

    scores = result.get("emotion") or {}
    if scores:
        return choose_model_emotion(scores)

    return normalize_emotion(result.get("dominant_emotion")), "dominant only"


def detect_with_haar(face_gray):
    face_gray = cv2.equalizeHist(face_gray)
    height, width = face_gray.shape[:2]

    lower_face = face_gray[int(height * 0.45):height, :]
    smiles = smile_cascade.detectMultiScale(
        lower_face,
        scaleFactor=1.8,
        minNeighbors=38,
        minSize=(max(35, width // 4), max(16, height // 10)),
    )

    for (smile_x, _, smile_w, _) in smiles:
        smile_center = smile_x + smile_w / 2
        if width * 0.25 <= smile_center <= width * 0.75:
            return "happy", "haar smile"

    upper_face = face_gray[:int(height * 0.55), :]
    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(max(14, width // 10), max(10, height // 12)),
    )

    if len(eyes) >= 2:
        eye_heights = sorted([eye_height for (_, _, _, eye_height) in eyes])[:2]
        eye_openness = sum(eye_heights) / (2 * height)

        if eye_openness < 0.09:
            return "angry", "haar narrow eyes"

    return "neutral", "haar neutral"


def detect_emotion(frame):
    global frame_count, last_model_emotion

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    raw_emotion = "neutral"
    stable_emotion = smoother.current

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        frame_count += 1

        if personal_model is not None:
            raw_emotion, reason = personal_model.predict(face_gray)
        elif DeepFace is not None and frame_count % MODEL_INTERVAL_FRAMES == 1:
            last_model_emotion = detect_with_deepface(face)
            raw_emotion, reason = last_model_emotion or detect_with_haar(face_gray)
        else:
            raw_emotion, reason = last_model_emotion or detect_with_haar(face_gray)

        stable_emotion = smoother.update(raw_emotion)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,stable_emotion,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)
        cv2.putText(frame,f"raw: {raw_emotion} ({reason})",(x,y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,(255,255,255),2)
        break

    if len(faces) == 0:
        stable_emotion = smoother.update("neutral")

    return frame, stable_emotion
