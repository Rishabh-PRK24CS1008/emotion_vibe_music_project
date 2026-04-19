from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from modules.personal_emotion_model import MODEL_PATH, extract_features


SAMPLES_PATH = Path("data") / "emotion_samples"
MIN_SAMPLES_PER_LABEL = 10


def load_samples():
    samples = defaultdict(list)

    for label_dir in sorted(SAMPLES_PATH.iterdir() if SAMPLES_PATH.exists() else []):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        for image_path in sorted(label_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            samples[label].append(extract_features(image))

    return samples


def main():
    samples = load_samples()
    if not samples:
        raise RuntimeError(
            "No samples found. Run collect_emotion_samples.py first."
        )

    labels = []
    centroids = []

    for label, features in sorted(samples.items()):
        print(f"{label}: {len(features)} samples")
        if len(features) < MIN_SAMPLES_PER_LABEL:
            print(f"  Skipping {label}; need at least {MIN_SAMPLES_PER_LABEL}.")
            continue

        labels.append(label)
        centroids.append(np.mean(np.vstack(features), axis=0))

    if len(labels) < 2:
        raise RuntimeError("Need at least two trained emotions.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        labels=np.array(labels),
        centroids=np.vstack(centroids).astype(np.float32),
    )
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
