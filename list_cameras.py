import os
import cv2


def probe_camera(index):
    backends = [cv2.CAP_ANY]
    if os.name == "nt":
        backends.insert(0, cv2.CAP_DSHOW)

    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            success, frame = camera.read()
            camera.release()
            if success and frame is not None:
                height, width = frame.shape[:2]
                return True, f"{width}x{height}"
        camera.release()

    return False, ""


if __name__ == "__main__":
    print("Available OpenCV camera indexes:")
    found = False

    for index in range(10):
        available, resolution = probe_camera(index)
        if available:
            found = True
            print(f"  {index}: working ({resolution})")

    if not found:
        print("  No working cameras found.")
