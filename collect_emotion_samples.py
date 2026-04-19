import argparse
from pathlib import Path

import cv2


LABEL_KEYS = {
    ord("h"): "happy",
    ord("s"): "sad",
    ord("a"): "angry",
    ord("n"): "neutral",
}


def open_camera(index):
    backends = [cv2.CAP_ANY]
    backends.insert(0, cv2.CAP_DSHOW)

    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            return camera
        camera.release()

    raise RuntimeError(f"Could not open camera index {index}")


def next_sample_path(label):
    folder = Path("data") / "emotion_samples" / label
    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob("*.png"))
    return folder / f"{label}_{len(existing) + 1:04d}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = open_camera(args.camera)

    print("Show one expression at a time, then press:")
    print("  h = happy")
    print("  s = sad")
    print("  a = angry")
    print("  n = neutral")
    print("  q = quit")
    print("Capture at least 25-40 samples per emotion for a useful model.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_crop = None
        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            break

        cv2.putText(
            frame,
            "h happy | s sad | a angry | n neutral | q quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Collect emotion samples", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key in LABEL_KEYS:
            if face_crop is None:
                print("No face detected. Try again.")
                continue

            label = LABEL_KEYS[key]
            path = next_sample_path(label)
            cv2.imwrite(str(path), face_crop)
            print(f"Saved {path}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
